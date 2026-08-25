"""
AST utilities for the Vulkan CTS extractor.

The implementation in this file is intentionally AST-only:
- no regex fallbacks for class/function discovery
- no line-by-line string parsing for module registration
- source snippets always come from AST source ranges
"""

import logging
import os
import xml.etree.ElementTree as ET
import gc
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import clang.cindex


class ASTUtils:
    """Clang AST helpers for Vulkan CTS source files."""

    _MSBUILD_NS = {"msb": "http://schemas.microsoft.com/developer/msbuild/2003"}
    _METHOD_KINDS = {
        clang.cindex.CursorKind.CXX_METHOD,
        clang.cindex.CursorKind.CONSTRUCTOR,
        clang.cindex.CursorKind.DESTRUCTOR,
    }
    _MODULE_CALL_NAMES = {"addRootChild"}
    _FUNCTION_CALL_NAMES = {"addFunctionCase", "addFunctionCaseWithPrograms"}
    _FUNCTION_WRAPPER_NAMES = {"addFunctionCaseInNewSubgroup"}
    _FUNCTION_FACTORY_NAMES = {"createFunctionCase", "createFunctionCaseWithPrograms"}
    _GROUP_FACTORY_NAMES = {"createTestGroup", "addTestGroup"}
    _CLASS_CALL_NAMES = {"addChild"}

    def __init__(
        self,
        project_root: Optional[Path] = None,
        build_root: Optional[Path] = None,
        preferred_configurations: Optional[List[str]] = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(project_root).resolve() if project_root else None
        self.build_root = Path(build_root).resolve() if build_root else (
            self.project_root / "build" if self.project_root else None
        )
        self.preferred_configurations = preferred_configurations or [
            "Debug|x64",
            "RelWithDebInfo|x64",
            "Release|x64",
            "MinSizeRel|x64",
        ]

        self._configured_libclang: Optional[Path] = None
        self._vcxproj_cache: Dict[str, Dict] = {}
        self._compile_args_cache: Dict[str, List[str]] = {}
        self._source_cache: Dict[str, str] = {}
        self._analysis_cache: Dict[str, Dict] = {}
        self._system_include_args: Optional[List[str]] = None
        self._clang_builtin_include_args: Optional[List[str]] = None
        self._forced_include_args: Optional[List[str]] = None

        self._setup_libclang()
        self.index = clang.cindex.Index.create()

    def _setup_libclang(self):
        # Allow an explicit override first, then inspect the installed Python
        # package before falling back to conventional system locations.
        candidate_paths = []
        _env_libclang = os.environ.get("LIBCLANG_PATH")
        if _env_libclang:
            candidate_paths.append(Path(_env_libclang))
        clang_package_dir = Path(clang.cindex.__file__).resolve().parent
        candidate_paths.extend(
            clang_package_dir / "native" / name
            for name in ("libclang.dll", "libclang.so", "libclang.dylib")
        )
        candidate_paths += [
            Path(r"C:\Program Files\LLVM\bin\libclang.dll"),
            Path(r"C:\LLVM\bin\libclang.dll"),
        ]

        for candidate in candidate_paths:
            if not candidate.exists():
                continue
            try:
                clang.cindex.Config.set_library_file(str(candidate))
                clang.cindex.Index.create()
                self._configured_libclang = candidate
                self.logger.info("libclang configured: %s", candidate)
                return
            except Exception:
                continue

        try:
            clang.cindex.Index.create()
            self.logger.info("libclang found automatically")
            return
        except clang.cindex.LibclangError:
            pass

        raise RuntimeError("Unable to locate a working libclang.dll")

    def clear_runtime_caches(self, *, keep_compile_args: bool = True):
        """
        Release per-file runtime caches to reduce long-running extractor memory/handle pressure.

        This is safe to call between modules after outputs are already materialized.
        """
        self._analysis_cache.clear()
        self._source_cache.clear()
        if not keep_compile_args:
            self._compile_args_cache.clear()
        gc.collect()

    def parse_file(self, file_path: Path) -> Optional[clang.cindex.TranslationUnit]:
        file_path = Path(file_path).resolve()
        polluted_env_vars = ("INCLUDE", "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH")
        saved_env = {name: os.environ.get(name) for name in polluted_env_vars}
        try:
            for name in polluted_env_vars:
                os.environ.pop(name, None)
            args = self._get_compile_args(file_path)
            options = (
                clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
                | clang.cindex.TranslationUnit.PARSE_INCOMPLETE
            )
            return self.index.parse(str(file_path), args=args, options=options)
        except Exception as exc:
            self.logger.error("Failed to parse %s: %s", file_path, exc)
            return None
        finally:
            for name, value in saved_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def extract_add_root_child_calls(self, file_path: Path) -> List[Dict]:
        analysis = self._analyze_file(file_path)
        return list(analysis["module_registrations"])

    def find_classes(self, file_path: Path, target_base_classes: Optional[List[str]] = None) -> List[Dict]:
        filter_by_base = target_base_classes is not None
        if target_base_classes is None:
            target_base_classes = []
        analysis = self._analyze_file(file_path)

        classes: List[Dict] = []
        for class_info in analysis["classes"].values():
            if filter_by_base and not self._matches_base_classes(class_info["base_classes"], target_base_classes):
                continue

            related_instance = self._find_associated_test_instance(class_info, analysis)
            classes.append(
                {
                    "name": class_info["name"],
                    "qualified_name": class_info["qualified_name"],
                    "namespace": class_info["namespace"],
                    "decl_kind": class_info.get("decl_kind", "class"),
                    "usr": class_info["usr"],
                    "file_path": class_info["file_path"],
                    "line_start": class_info["line_start"],
                    "line_end": class_info["line_end"],
                    "base_classes": list(class_info["base_classes"]),
                    "code": class_info["code"],
                    "registrations": list(class_info.get("registrations", [])),
                    "associated_test_instance": related_instance["qualified_name"] if related_instance else None,
                }
            )

        classes.sort(key=lambda item: (item["line_start"], item["name"]))
        return classes

    def find_test_functions(self, file_path: Path, registered_only: bool = False) -> List[Dict]:
        analysis = self._analyze_file(file_path)
        functions: List[Dict] = []

        for func_info in analysis["functions"].values():
            if not func_info["is_test_function"]:
                continue

            registrations = analysis["function_registrations"].get(func_info["usr"], [])
            if registered_only and not registrations:
                continue

            functions.append(
                {
                    "name": func_info["name"],
                    "qualified_name": func_info["qualified_name"],
                    "namespace": func_info["namespace"],
                    "usr": func_info["usr"],
                    "signature": func_info["signature"],
                    "return_type": func_info["return_type"],
                    "file_path": func_info["file_path"],
                    "line_start": func_info["line_start"],
                    "line_end": func_info["line_end"],
                    "code": func_info["code"],
                    "registrations": list(registrations),
                }
            )

        functions.sort(key=lambda item: (item["line_start"], item["name"]))
        return functions

    def build_complete_class_test(
        self,
        file_path: Path,
        class_info: Dict,
        extra_files: Optional[List[Path]] = None,
    ) -> str:
        primary_analysis = self._analyze_file(file_path)
        record = self._resolve_class_record(
            primary_analysis,
            class_info.get("qualified_name") or class_info.get("name"),
        )
        if not record:
            return class_info.get("code", "")

        analyses = [primary_analysis]
        seen_files = {self._normalize_path(file_path)}
        for extra_file in extra_files or []:
            normalized = self._normalize_path(extra_file)
            if normalized in seen_files:
                continue
            seen_files.add(normalized)
            analyses.append(self._analyze_file(extra_file))

        segments = [record["code"].rstrip()]

        out_of_line_methods = self._get_out_of_line_methods_from_analyses(record, analyses)
        if out_of_line_methods:
            method_text = "\n\n".join(method["code"].rstrip() for method in out_of_line_methods if method["code"].strip())
            if method_text:
                segments.append("// === Methods ===\n\n" + method_text)

        instance_record = self._find_associated_test_instance_from_analyses(record, analyses)
        if instance_record:
            segments.append("// === TestInstance ===\n\n" + instance_record["code"].rstrip())
            instance_methods = self._get_out_of_line_methods_from_analyses(instance_record, analyses)
            if instance_methods:
                method_text = "\n\n".join(
                    method["code"].rstrip() for method in instance_methods if method["code"].strip()
                )
                if method_text:
                    segments.append("// === TestInstance Methods ===\n\n" + method_text)

        return "\n\n".join(segment for segment in segments if segment).strip()

    def get_class_base_classes(self, file_path: Path, class_name: str) -> List[str]:
        analysis = self._analyze_file(file_path)
        record = self._resolve_class_record(analysis, class_name)
        return list(record["base_classes"]) if record else []

    def _analyze_file(self, file_path: Path) -> Dict:
        file_path = Path(file_path).resolve()
        cache_key = self._normalize_path(file_path)
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        analysis = {
            "translation_unit": None,
            "classes": {},
            "class_name_index": {},
            "qualified_name_index": {},
            "template_fields_by_name": {},
            "methods_by_parent": {},
            "functions": {},
            "function_defs_by_name": {},
            "var_decls_by_name": {},
            "function_registrations": {},
            "module_registrations": [],
        }

        tu = self.parse_file(file_path)
        analysis["translation_unit"] = tu
        if not tu:
            self._analysis_cache[cache_key] = analysis
            return analysis

        for cursor in tu.cursor.walk_preorder():
            if not self._is_same_file_cursor(cursor, file_path):
                continue

            if cursor.kind in {clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL}:
                if cursor.is_definition() and cursor.spelling:
                    class_info = self._build_class_info(cursor, file_path)
                    analysis["classes"][class_info["usr"]] = class_info
                    analysis["class_name_index"].setdefault(class_info["name"], set()).add(class_info["usr"])
                    analysis["qualified_name_index"][class_info["qualified_name"]] = class_info["usr"]
                continue

            if cursor.kind == clang.cindex.CursorKind.CLASS_TEMPLATE and cursor.spelling:
                field_names = [child.spelling for child in cursor.get_children() if child.kind == clang.cindex.CursorKind.FIELD_DECL]
                if field_names:
                    analysis["template_fields_by_name"][cursor.spelling] = field_names
                # A class template can itself be a test class (e.g. OpenGL CTS
                # `template<typename T> class LimitCase : public deqp::TestCase`,
                # registered via `addChild(new LimitCase<GLint>(...))`, and the
                # 62 ArrayOfArrays templates in gles31). Treat the template's
                # primary definition as a class so it enters the candidate set;
                # downstream registration/inheritance gates still filter non-tests.
                if cursor.is_definition():
                    class_info = self._build_class_info(cursor, file_path)
                    analysis["classes"].setdefault(class_info["usr"], class_info)
                    analysis["class_name_index"].setdefault(class_info["name"], set()).add(class_info["usr"])
                    analysis["qualified_name_index"].setdefault(class_info["qualified_name"], class_info["usr"])
                continue

            if cursor.kind in self._METHOD_KINDS and cursor.is_definition():
                method_info = self._build_method_info(cursor, file_path)
                parent = cursor.semantic_parent
                parent_usr = parent.get_usr() if parent else None
                if parent_usr:
                    analysis["methods_by_parent"].setdefault(parent_usr, []).append(method_info)
                continue

            function_decl = self._extract_function_decl_cursor(cursor)
            if function_decl is not None and function_decl.spelling:
                func_info = self._build_function_info(function_decl, file_path)
                analysis["functions"][func_info["usr"]] = func_info
                defs = analysis["function_defs_by_name"].setdefault(func_info["name"], [])
                if all(existing["usr"] != func_info["usr"] for existing in defs):
                    defs.append(func_info)

            if cursor.kind == clang.cindex.CursorKind.VAR_DECL and cursor.spelling:
                analysis["var_decls_by_name"].setdefault(cursor.spelling, []).append(cursor)

            if cursor.kind == clang.cindex.CursorKind.CALL_EXPR:
                self._collect_registration(cursor, file_path, analysis)

        for class_info in analysis["classes"].values():
            class_info["registrations"] = list(analysis.get("class_registrations", {}).get(class_info["usr"], []))

        for method_list in analysis["methods_by_parent"].values():
            method_list.sort(key=lambda item: (item["line_start"], item["name"]))

        for registration_list in analysis["function_registrations"].values():
            registration_list.sort(key=lambda item: item["line"])

        analysis["module_registrations"].sort(key=lambda item: item["line"])
        self._analysis_cache[cache_key] = analysis
        return analysis

    def _build_class_info(self, cursor, file_path: Path) -> Dict:
        base_classes = self._get_base_classes(cursor)
        qualified_name = self._get_qualified_name(cursor)
        namespace = self._get_namespace(cursor)
        return {
            "usr": cursor.get_usr(),
            "name": cursor.spelling,
            "qualified_name": qualified_name,
            "namespace": namespace,
            "decl_kind": "struct" if cursor.kind == clang.cindex.CursorKind.STRUCT_DECL else "class",
            "file_path": str(file_path),
            "line_start": cursor.extent.start.line,
            "line_end": cursor.extent.end.line,
            "base_classes": base_classes,
            "code": self._cursor_text(file_path, cursor),
            "cursor": cursor,
            "is_test_case": self._matches_base_classes(base_classes, ["TestCase"]),
            "is_test_instance": self._matches_base_classes(base_classes, ["TestInstance"]),
        }

    def _build_method_info(self, cursor, file_path: Path) -> Dict:
        parent = cursor.semantic_parent
        return {
            "usr": cursor.get_usr(),
            "name": cursor.spelling,
            "qualified_name": self._get_qualified_name(cursor),
            "file_path": str(file_path),
            "parent_usr": parent.get_usr() if parent else None,
            "parent_name": parent.spelling if parent else "",
            "parent_qualified_name": self._get_qualified_name(parent) if parent and parent.spelling else "",
            "line_start": cursor.extent.start.line,
            "line_end": cursor.extent.end.line,
            "code": self._cursor_text(file_path, cursor),
            "cursor": cursor,
        }

    def _extract_function_decl_cursor(self, cursor):
        if cursor.kind in {clang.cindex.CursorKind.FUNCTION_DECL, clang.cindex.CursorKind.FUNCTION_TEMPLATE}:
            return cursor
        return None

    def _get_function_parameters(self, cursor):
        try:
            params = list(cursor.get_arguments() or [])
        except Exception:
            params = []
        if params:
            return params
        return [child for child in cursor.get_children() if child.kind == clang.cindex.CursorKind.PARM_DECL]

    def _build_function_info(self, cursor, file_path: Path) -> Dict:
        params = self._get_function_parameters(cursor)
        first_param_type = params[0].type.spelling if params else ""
        return_type = cursor.result_type.spelling
        return {
            "usr": cursor.get_usr(),
            "name": cursor.spelling,
            "qualified_name": self._get_qualified_name(cursor),
            "namespace": self._get_namespace(cursor),
            "signature": cursor.displayname,
            "return_type": return_type,
            "file_path": str(file_path),
            "line_start": cursor.extent.start.line,
            "line_end": cursor.extent.end.line,
            "code": self._cursor_text(file_path, cursor),
            "cursor": cursor,
            "is_test_function": "TestStatus" in return_type and "Context" in first_param_type,
        }

    def _collect_registration(self, call_cursor, file_path: Path, analysis: Dict):
        call_name = self._extract_call_name(call_cursor)
        if not call_name:
            return

        if call_name in self._MODULE_CALL_NAMES:
            module_info = self._parse_module_registration(call_cursor, file_path)
            if module_info:
                analysis["module_registrations"].append(module_info)
            return

        if call_name in self._FUNCTION_CALL_NAMES | self._FUNCTION_WRAPPER_NAMES:
            function_cursors = self._resolve_registered_functions(call_cursor, file_path, analysis=analysis)
            if not function_cursors:
                return

            registration = {
                "kind": call_name,
                "line": call_cursor.extent.start.line,
                "call": self._cursor_text(file_path, call_cursor),
            }
            for function_cursor in function_cursors:
                analysis["function_registrations"].setdefault(function_cursor.get_usr(), []).append(registration)
            return

        if call_name in self._CLASS_CALL_NAMES:
            registration = {
                "kind": call_name,
                "line": call_cursor.extent.start.line,
                "call": self._cursor_text(file_path, call_cursor),
            }

            class_record = self._resolve_registered_class(call_cursor, analysis)
            if class_record:
                analysis.setdefault("class_registrations", {}).setdefault(class_record["usr"], []).append(registration)
                return

            function_cursors = self._resolve_group_factory_registration(call_cursor, file_path, analysis)
            if not function_cursors:
                function_cursors = self._resolve_registered_functions(call_cursor, file_path, analysis=analysis)
            for function_cursor in function_cursors:
                analysis["function_registrations"].setdefault(function_cursor.get_usr(), []).append(registration)

    def _parse_module_registration(self, call_cursor, file_path: Path) -> Optional[Dict]:
        registered_name = None
        for node in self._iter_descendants(call_cursor):
            if node.kind == clang.cindex.CursorKind.STRING_LITERAL:
                registered_name = self._string_literal_value(node, file_path)
                if registered_name:
                    break

        function_cursor = None
        for node in self._iter_descendants(call_cursor):
            referenced = getattr(node, "referenced", None)
            if referenced and referenced.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                if referenced.spelling == "addRootChild":
                    continue
                function_cursor = referenced

        if not registered_name:
            return None

        container_name = self._container_name(function_cursor.semantic_parent) if function_cursor else None
        namespace = registered_name if not container_name or container_name == "vkt" else container_name

        return {
            "registered_name": registered_name,
            "namespace": namespace,
            "line": call_cursor.extent.start.line,
            "call": self._cursor_text(file_path, call_cursor),
        }

    def _resolve_registered_functions(
        self,
        call_cursor,
        file_path: Path,
        parameter_bindings: Optional[Dict[str, object]] = None,
        visited: Optional[Set] = None,
        depth: int = 0,
        analysis: Optional[Dict] = None,
    ) -> List[object]:
        if depth > 8:
            return []

        results: List[object] = []
        seen_usrs = set()
        if visited is None:
            visited = set()

        def add_result(cursor_obj):
            if not cursor_obj:
                return
            usr = cursor_obj.get_usr()
            if usr in seen_usrs:
                return
            seen_usrs.add(usr)
            results.append(cursor_obj)

        direct = self._resolve_function_reference_from_expr(
            call_cursor,
            file_path,
            parameter_bindings=parameter_bindings,
            visited=visited,
            analysis=analysis,
        )
        add_result(direct)

        if self._extract_call_name(call_cursor) in self._CLASS_CALL_NAMES:
            for node in self._iter_descendants(call_cursor):
                if node.kind != clang.cindex.CursorKind.CALL_EXPR or node == call_cursor:
                    continue
                nested = self._resolve_registered_functions(
                    node,
                    file_path,
                    parameter_bindings=parameter_bindings,
                    visited=visited,
                    depth=depth + 1,
                    analysis=analysis,
                )
                for cursor_obj in nested:
                    add_result(cursor_obj)

        callee_info = self._resolve_called_function_definition(call_cursor, file_path, analysis=analysis)
        if not callee_info:
            return results

        callee_cursor = callee_info["cursor"]
        callee_usr = callee_info["usr"]
        visit_key = (callee_usr, call_cursor.extent.start.line, depth, 'multi')
        if visit_key in visited:
            return results
        visited.add(visit_key)

        nested_bindings = self._build_parameter_bindings(callee_cursor, call_cursor, parameter_bindings)
        callee_file = Path(callee_info["file_path"])

        for node in callee_cursor.walk_preorder():
            if node.kind != clang.cindex.CursorKind.CALL_EXPR:
                continue
            nested_name = self._get_call_callee_name(node) or ""
            if not self._extract_call_name(node) and not nested_name.startswith(("create", "add", "make")):
                continue
            nested = self._resolve_registered_functions(
                node,
                callee_file,
                parameter_bindings=nested_bindings,
                visited=visited,
                depth=depth + 1,
                analysis=analysis if self._normalize_path(callee_file) == self._normalize_path(file_path) else None,
            )
            for cursor_obj in nested:
                add_result(cursor_obj)
        return results

    def _resolve_registered_function(
        self,
        call_cursor,
        file_path: Path,
        parameter_bindings: Optional[Dict[str, object]] = None,
        visited: Optional[Set] = None,
        depth: int = 0,
        analysis: Optional[Dict] = None,
    ):
        results = self._resolve_registered_functions(
            call_cursor,
            file_path,
            parameter_bindings=parameter_bindings,
            visited=visited,
            depth=depth,
            analysis=analysis,
        )
        return results[0] if results else None

    def _resolve_group_factory_registration(self, call_cursor, file_path: Path, analysis: Dict) -> List[object]:
        for node in self._iter_descendants(call_cursor):
            if node.kind != clang.cindex.CursorKind.CALL_EXPR:
                continue
            if self._get_call_callee_name(node) not in self._GROUP_FACTORY_NAMES:
                continue
            return self._resolve_group_factory_functions(node, file_path, analysis=analysis)
        return []

    def _resolve_group_factory_functions(
        self,
        call_cursor,
        file_path: Path,
        parameter_bindings: Optional[Dict[str, object]] = None,
        visited: Optional[Set] = None,
        depth: int = 0,
        analysis: Optional[Dict] = None,
    ) -> List[object]:
        call_name = self._get_call_callee_name(call_cursor)
        if call_name not in self._GROUP_FACTORY_NAMES or depth > 8:
            return []

        args = self._get_call_arguments(call_cursor)
        if len(args) < 3:
            return []

        callback_expr = self._substitute_bound_expr(args[2], parameter_bindings)
        callback_cursor = self._resolve_callback_function_from_expr(
            callback_expr,
            file_path,
            analysis=analysis,
        )
        if not callback_cursor:
            return []

        callback_params = self._get_function_parameters(callback_cursor)
        callback_bindings: Dict[str, object] = {}
        for param, arg in zip(callback_params[1:], args[3:]):
            callback_bindings[param.get_usr()] = self._substitute_bound_expr(arg, parameter_bindings)

        if visited is None:
            visited = set()
        callback_visit_key = ("group_factory", callback_cursor.get_usr(), call_cursor.extent.start.line)
        if callback_visit_key in visited:
            return []
        visited.add(callback_visit_key)

        callback_file_path = getattr(callback_cursor.location, "file", None)
        callback_file = Path(str(callback_file_path)).resolve() if callback_file_path else file_path
        callback_analysis = analysis if self._normalize_path(callback_file) == self._normalize_path(file_path) else None

        results: List[object] = []
        seen_usrs: Set[str] = set()
        for node in callback_cursor.walk_preorder():
            if node.kind != clang.cindex.CursorKind.CALL_EXPR:
                continue
            nested_name = self._get_call_callee_name(node) or ""
            if nested_name not in (
                self._FUNCTION_CALL_NAMES
                | self._FUNCTION_WRAPPER_NAMES
                | self._FUNCTION_FACTORY_NAMES
                | self._CLASS_CALL_NAMES
            ):
                continue
            nested = self._resolve_registered_functions(
                node,
                callback_file,
                parameter_bindings=callback_bindings,
                visited=visited,
                depth=depth + 1,
                analysis=callback_analysis,
            )
            for cursor_obj in nested:
                usr = cursor_obj.get_usr()
                if usr in seen_usrs:
                    continue
                seen_usrs.add(usr)
                results.append(cursor_obj)
        return results

    def _resolve_function_reference_from_expr(
        self,
        cursor,
        file_path: Path,
        parameter_bindings: Optional[Dict[str, object]] = None,
        visited: Optional[Set] = None,
        analysis: Optional[Dict] = None,
    ):
        if self._is_bound_expr_wrapper(cursor):
            target = self._resolve_function_from_bound_member_chain(
                cursor["expr"],
                cursor["suffix"],
                Path(cursor["file_path"]),
                parameter_bindings=parameter_bindings,
                analysis=analysis,
            )
            if target:
                return target
            cursor = cursor["expr"]

        preferred = []
        fallback = []

        for node in [cursor, *self._iter_descendants(cursor)]:
            referenced = getattr(node, "referenced", None)
            if not referenced:
                continue

            if referenced.kind == clang.cindex.CursorKind.PARM_DECL and parameter_bindings:
                bound_expr = parameter_bindings.get(referenced.get_usr())
                if bound_expr is not None:
                    bound_target = self._resolve_function_reference_from_expr(
                        bound_expr,
                        file_path,
                        parameter_bindings=parameter_bindings,
                        visited=visited,
                        analysis=analysis,
                    )
                    if bound_target:
                        return bound_target

                    full_text = self._cursor_text(file_path, cursor).strip()
                    suffix = self._extract_member_suffix(full_text, referenced.spelling)
                    if suffix:
                        chain_target = self._resolve_function_from_bound_member_chain(
                            bound_expr,
                            suffix,
                            file_path,
                            parameter_bindings=parameter_bindings,
                            analysis=analysis,
                        )
                        if chain_target:
                            return chain_target
                continue

            if referenced.kind != clang.cindex.CursorKind.FUNCTION_DECL:
                continue
            if referenced.spelling in (
                self._FUNCTION_CALL_NAMES
                | self._FUNCTION_WRAPPER_NAMES
                | self._FUNCTION_FACTORY_NAMES
            ):
                continue

            target = referenced.get_definition() or referenced
            if "TestStatus" not in target.result_type.spelling:
                continue

            if self._is_same_file_cursor(target, file_path):
                preferred.append(target)
            else:
                fallback.append(target)

        return preferred[-1] if preferred else (fallback[-1] if fallback else None)

    def _resolve_callback_function_from_expr(
        self,
        cursor,
        file_path: Path,
        analysis: Optional[Dict] = None,
    ):
        preferred = []
        fallback = []

        for node in [cursor, *self._iter_descendants(cursor)]:
            referenced = getattr(node, "referenced", None)
            if not referenced or referenced.kind not in {
                clang.cindex.CursorKind.FUNCTION_DECL,
                clang.cindex.CursorKind.FUNCTION_TEMPLATE,
            }:
                continue

            target = referenced.get_definition() or referenced
            if self._is_same_file_cursor(target, file_path):
                preferred.append(target)
            else:
                fallback.append(target)

        if preferred or fallback:
            return preferred[-1] if preferred else fallback[-1]

        call_name = self._get_call_callee_name(cursor)
        if not call_name:
            return None

        active_analysis = analysis if analysis is not None else self._analyze_file(file_path)
        candidates = list(active_analysis.get("function_defs_by_name", {}).get(call_name, []))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item["line_start"], item["qualified_name"]))
        return candidates[0]["cursor"]

    def _resolve_called_function_definition(self, call_cursor, file_path: Path, analysis: Optional[Dict] = None) -> Optional[Dict]:
        call_name = self._get_call_callee_name(call_cursor)
        if not call_name:
            return None

        active_analysis = analysis if analysis is not None else self._analyze_file(file_path)
        candidates = list(active_analysis.get("function_defs_by_name", {}).get(call_name, []))

        referenced = getattr(call_cursor, "referenced", None)
        referenced_file: Optional[Path] = None
        referenced_line = 0
        if referenced and getattr(referenced, "location", None) and getattr(referenced.location, "file", None):
            referenced_file = Path(str(referenced.location.file)).resolve()
            referenced_line = referenced.location.line or 0

        if not candidates and referenced_file and call_name in self._GROUP_FACTORY_NAMES:
            referenced_analysis = self._analyze_file(referenced_file)
            candidates = list(referenced_analysis.get("function_defs_by_name", {}).get(call_name, []))

        if not candidates:
            return None

        argc = len(self._get_call_arguments(call_cursor))
        exact = [item for item in candidates if len(list(item["cursor"].get_arguments() or [])) == argc]
        if exact:
            candidates = exact

        if referenced_line:
            line_matched = [item for item in candidates if item["line_start"] == referenced_line]
            if line_matched:
                candidates = line_matched

        candidates.sort(
            key=lambda item: (
                0
                if referenced_file and self._normalize_path(Path(item["file_path"])) == self._normalize_path(referenced_file)
                else 1,
                abs(item["line_start"] - referenced_line) if referenced_line else item["line_start"],
                item["qualified_name"],
            )
        )
        return candidates[0]

    def _build_parameter_bindings(
        self,
        callee_cursor,
        call_cursor,
        outer_bindings: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        bindings: Dict[str, object] = {}
        params = self._get_function_parameters(callee_cursor)
        args = self._get_call_arguments(call_cursor)
        for param, arg in zip(params, args):
            bindings[param.get_usr()] = self._substitute_bound_expr(arg, outer_bindings)
        return bindings

    def _substitute_bound_expr(self, expr_cursor, bindings: Optional[Dict[str, object]] = None):
        if not bindings:
            return expr_cursor
        if self._is_bound_expr_wrapper(expr_cursor):
            return expr_cursor

        referenced = getattr(expr_cursor, "referenced", None)
        if referenced and referenced.kind == clang.cindex.CursorKind.PARM_DECL:
            return bindings.get(referenced.get_usr(), expr_cursor)

        location = getattr(expr_cursor, "location", None)
        location_file = getattr(location, "file", None)
        scope_file = Path(location_file.name) if location_file else None
        expr_text = self._cursor_text(scope_file, expr_cursor).strip() if scope_file else ""
        if not expr_text:
            return expr_cursor

        for node in [expr_cursor, *self._iter_descendants(expr_cursor)]:
            node_ref = getattr(node, "referenced", None)
            if not node_ref or node_ref.kind != clang.cindex.CursorKind.PARM_DECL:
                continue
            bound = bindings.get(node_ref.get_usr())
            if bound is None:
                continue
            suffix = self._extract_member_suffix(expr_text, node_ref.spelling)
            if suffix:
                return {"expr": bound, "suffix": suffix, "file_path": str(scope_file) if scope_file else ""}
            if expr_text == node_ref.spelling:
                return bound
        return expr_cursor

    def _get_call_arguments(self, call_cursor) -> List[object]:
        try:
            return list(call_cursor.get_arguments() or [])
        except Exception:
            children = list(call_cursor.get_children())
            return children[1:] if len(children) > 1 else []

    def _resolve_registered_class(self, call_cursor, analysis: Dict) -> Optional[Dict]:
        for node in self._iter_descendants(call_cursor):
            if node.kind != clang.cindex.CursorKind.CXX_NEW_EXPR:
                continue

            referenced = getattr(node, "referenced", None)
            class_record = self._resolve_class_record_from_cursor(analysis, referenced)
            if class_record:
                return class_record

            target_name = self._normalize_type_name(node.type.spelling)
            class_record = self._resolve_class_record(analysis, target_name)
            if class_record:
                return class_record

            for descendant in self._iter_descendants(node):
                referenced = getattr(descendant, "referenced", None)
                class_record = self._resolve_class_record_from_cursor(analysis, referenced)
                if class_record:
                    return class_record
                if descendant.kind == clang.cindex.CursorKind.TYPE_REF and descendant.spelling:
                    class_record = self._resolve_class_record(analysis, descendant.spelling)
                    if class_record:
                        return class_record

        return None

    def _find_associated_test_instance(self, class_info: Dict, analysis: Dict) -> Optional[Dict]:
        for method in analysis["methods_by_parent"].get(class_info["usr"], []):
            if method["name"] != "createInstance":
                continue

            cursor = method["cursor"]
            for node in cursor.walk_preorder():
                if node.kind != clang.cindex.CursorKind.CXX_NEW_EXPR:
                    continue

                referenced = getattr(node, "referenced", None)
                instance_record = self._resolve_class_record_from_cursor(analysis, referenced)
                if instance_record and instance_record["is_test_instance"]:
                    return instance_record

                target_name = self._normalize_type_name(node.type.spelling)
                instance_record = self._resolve_class_record(analysis, target_name)
                if instance_record and instance_record["is_test_instance"]:
                    return instance_record

                for descendant in self._iter_descendants(node):
                    referenced = getattr(descendant, "referenced", None)
                    instance_record = self._resolve_class_record_from_cursor(analysis, referenced)
                    if instance_record and instance_record["is_test_instance"]:
                        return instance_record
                    if descendant.kind == clang.cindex.CursorKind.TYPE_REF and descendant.spelling:
                        instance_record = self._resolve_class_record(analysis, descendant.spelling)
                        if instance_record and instance_record["is_test_instance"]:
                            return instance_record

        return None

    def _find_associated_test_instance_from_analyses(self, class_info: Dict, analyses: List[Dict]) -> Optional[Dict]:
        for analysis in analyses:
            instance_record = self._find_associated_test_instance(class_info, analysis)
            if instance_record:
                return instance_record
        return None

    def _get_out_of_line_methods(self, class_info: Dict, analysis: Dict) -> List[Dict]:
        methods = analysis["methods_by_parent"].get(class_info["usr"], [])
        return [
            method
            for method in methods
            if method["line_start"] < class_info["line_start"] or method["line_end"] > class_info["line_end"]
        ]

    def _get_out_of_line_methods_from_analyses(self, class_info: Dict, analyses: List[Dict]) -> List[Dict]:
        methods: List[Dict] = []
        seen = set()
        class_file = self._normalize_path(class_info["file_path"])

        for analysis in analyses:
            methods_for_analysis = list(analysis["methods_by_parent"].get(class_info["usr"], []))
            if not methods_for_analysis and class_info.get("qualified_name"):
                for method_list in analysis["methods_by_parent"].values():
                    for method in method_list:
                        if self._method_belongs_to_class(method, class_info):
                            methods_for_analysis.append(method)

            for method in methods_for_analysis:
                method_file = self._normalize_path(method.get("file_path", class_info["file_path"]))
                is_out_of_line = (
                    method_file != class_file
                    or method["line_start"] < class_info["line_start"]
                    or method["line_end"] > class_info["line_end"]
                )
                if not is_out_of_line:
                    continue

                key = (
                    method.get("qualified_name"),
                    method.get("file_path"),
                    method["line_start"],
                    method["line_end"],
                )
                if key in seen:
                    continue
                seen.add(key)
                methods.append(method)

        methods.sort(
            key=lambda item: (
                self._normalize_path(item.get("file_path", "")),
                item["line_start"],
                item["name"],
            )
        )
        return methods

    def _get_base_classes(self, class_cursor) -> List[str]:
        bases: List[str] = []
        for child in class_cursor.get_children():
            if child.kind == clang.cindex.CursorKind.CXX_BASE_SPECIFIER:
                bases.append(self._normalize_type_name(child.type.spelling))
        return bases

    def _matches_base_classes(self, bases: List[str], targets: List[str]) -> bool:
        for base_name in bases:
            for target in targets:
                if target in base_name:
                    return True
        return False

    def _resolve_class_record(self, analysis: Dict, class_name: Optional[str]) -> Optional[Dict]:
        if not class_name:
            return None

        normalized = self._normalize_type_name(class_name)
        direct_usr = analysis["qualified_name_index"].get(normalized)
        if direct_usr:
            return analysis["classes"].get(direct_usr)

        simple_name = normalized.split("::")[-1]
        candidates = analysis["class_name_index"].get(simple_name, set())
        if not candidates:
            return None

        records = [analysis["classes"][usr] for usr in candidates if usr in analysis["classes"]]
        records.sort(key=lambda item: (not item["is_test_case"], not item["is_test_instance"], item["qualified_name"]))
        return records[0] if records else None

    def _resolve_class_record_from_cursor(self, analysis: Dict, cursor) -> Optional[Dict]:
        if not cursor or cursor.kind not in {
            clang.cindex.CursorKind.CLASS_DECL,
            clang.cindex.CursorKind.STRUCT_DECL,
            clang.cindex.CursorKind.CLASS_TEMPLATE,
        }:
            return None

        target = cursor.get_definition() or cursor
        usr = target.get_usr()
        if usr in analysis["classes"]:
            return analysis["classes"][usr]

        qualified_name = self._get_qualified_name(target)
        return self._resolve_class_record(analysis, qualified_name or target.spelling)

    def _method_belongs_to_class(self, method: Dict, class_info: Dict) -> bool:
        if method.get("parent_usr") and method.get("parent_usr") == class_info.get("usr"):
            return True

        parent_qname = method.get("parent_qualified_name") or ""
        class_qname = class_info.get("qualified_name") or ""
        return bool(parent_qname and class_qname and parent_qname == class_qname)

    def _is_bound_expr_wrapper(self, value) -> bool:
        return isinstance(value, dict) and {"expr", "suffix", "file_path"}.issubset(value.keys())

    def _extract_member_suffix(self, expr_text: str, base_name: str) -> str:
        compact = "".join(expr_text.split())
        patterns = [f"{base_name}.", f"{base_name}->"]
        for pattern in patterns:
            index = compact.find(pattern)
            if index == -1:
                continue
            suffix = compact[index + len(base_name) :]
            if suffix.startswith('.') or suffix.startswith('->'):
                end_index = len(suffix)
                for separator in [',', ')']:
                    sep_index = suffix.find(separator)
                    if sep_index != -1:
                        end_index = min(end_index, sep_index)
                return suffix[:end_index]
        return ""

    def _resolve_function_from_bound_member_chain(
        self,
        bound_expr,
        suffix: str,
        file_path: Path,
        parameter_bindings: Optional[Dict[str, object]] = None,
        analysis: Optional[Dict] = None,
    ):
        if self._is_bound_expr_wrapper(bound_expr):
            combined_suffix = bound_expr["suffix"] + suffix
            return self._resolve_function_from_bound_member_chain(
                bound_expr["expr"],
                combined_suffix,
                Path(bound_expr["file_path"] or file_path),
                parameter_bindings=parameter_bindings,
                analysis=analysis,
            )

        bound_file = file_path
        location = getattr(bound_expr, "location", None)
        location_file = getattr(location, "file", None)
        if location_file:
            bound_file = Path(location_file.name)

        bound_text = "".join(self._cursor_text(bound_file, bound_expr).split())
        if not bound_text:
            return None

        combined = bound_text + suffix.replace("->", ".")
        combined = combined.replace("->", ".")

        param_refs = []
        referenced = getattr(bound_expr, "referenced", None)
        if referenced and referenced.kind == clang.cindex.CursorKind.PARM_DECL:
            param_refs.append(referenced)
        for node in self._iter_descendants(bound_expr):
            node_ref = getattr(node, "referenced", None)
            if node_ref and node_ref.kind == clang.cindex.CursorKind.PARM_DECL:
                param_refs.append(node_ref)

        if parameter_bindings:
            for param_ref in param_refs:
                next_bound = parameter_bindings.get(param_ref.get_usr())
                if next_bound is None:
                    continue
                next_suffix = self._extract_member_suffix(combined, param_ref.spelling)
                if next_suffix:
                    return self._resolve_function_from_bound_member_chain(
                        next_bound,
                        next_suffix,
                        bound_file,
                        parameter_bindings=parameter_bindings,
                        analysis=analysis,
                    )

        active_analysis = analysis if analysis is not None else self._analyze_file(bound_file)
        return self._resolve_function_from_member_chain_text(combined, active_analysis)

    def _resolve_function_from_member_chain_text(self, expr_text: str, analysis: Dict):
        compact = expr_text.replace("->", ".")
        parts = [part for part in compact.split(".") if part]
        if len(parts) < 2:
            return None

        var_candidates = analysis.get("var_decls_by_name", {}).get(parts[0], [])
        for var_cursor in var_candidates:
            target = self._resolve_function_from_var_decl(var_cursor, parts[1:], analysis)
            if target:
                return target
        return None

    def _resolve_function_from_var_decl(self, var_cursor, field_path: List[str], analysis: Dict):
        current_init = None
        for child in var_cursor.get_children():
            if child.kind == clang.cindex.CursorKind.INIT_LIST_EXPR:
                current_init = child
                break
        if current_init is None:
            return None

        type_cursor = var_cursor.type.get_declaration()
        remaining = list(field_path)
        while remaining and type_cursor:
            field_name = remaining.pop(0)
            field_cursors = [child for child in type_cursor.get_children() if child.kind == clang.cindex.CursorKind.FIELD_DECL]
            field_index = next((idx for idx, child in enumerate(field_cursors) if child.spelling == field_name), None)
            if field_index is None:
                template_fields = analysis.get("template_fields_by_name", {}).get(type_cursor.spelling, [])
                field_index = next((idx for idx, name in enumerate(template_fields) if name == field_name), None)
            if field_index is None:
                return None

            init_children = list(current_init.get_children())
            if field_index >= len(init_children):
                return None
            selected = init_children[field_index]
            if not remaining:
                return self._resolve_function_from_initializer(selected, analysis)

            current_init = selected
            if field_cursors and field_index < len(field_cursors):
                type_cursor = field_cursors[field_index].type.get_declaration()

        return None

    def _resolve_function_from_initializer(self, cursor, analysis: Dict):
        candidate_names: List[str] = []
        for node in cursor.walk_preorder():
            referenced = getattr(node, "referenced", None)
            if referenced and referenced.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                target = referenced.get_definition() or referenced
                if "TestStatus" in target.result_type.spelling:
                    return target
            if node.kind == clang.cindex.CursorKind.OVERLOADED_DECL_REF and node.spelling:
                candidate_names.append(node.spelling)
            elif node.kind == clang.cindex.CursorKind.DECL_REF_EXPR and node.spelling:
                candidate_names.append(node.spelling)

        callee_name = self._get_call_callee_name(cursor)
        if callee_name:
            candidate_names.append(callee_name)

        seen = set()
        for name in candidate_names:
            if not name or name in seen:
                continue
            seen.add(name)
            for candidate in analysis.get("function_defs_by_name", {}).get(name, []):
                target = candidate["cursor"]
                if "TestStatus" in target.result_type.spelling:
                    return target
        return None

    def _get_compile_args(self, file_path: Path) -> List[str]:
        file_path = Path(file_path).resolve()
        cache_key = self._normalize_path(file_path)
        if cache_key in self._compile_args_cache:
            return list(self._compile_args_cache[cache_key])

        args = [
            "-std=c++17",
            "-x",
            "c++",
            "--target=x86_64-pc-windows-msvc",
            "-fms-compatibility",
            "-fms-extensions",
        ]
        args.extend(self._get_clang_builtin_include_args())
        args.extend(self._get_system_include_args())
        args.extend(self._get_forced_include_args())

        vcxproj = self._find_vcxproj_for_file(file_path)
        if vcxproj:
            project_settings = self._load_vcxproj_settings(vcxproj)
            args.extend(project_settings["include_args"])
            args.extend(project_settings["define_args"])
        else:
            args.extend(self._get_default_project_include_args())
            args.extend(self._get_default_define_args())

        self._compile_args_cache[cache_key] = list(args)
        return args

    def _get_clang_builtin_include_args(self) -> List[str]:
        if self._clang_builtin_include_args is not None:
            return list(self._clang_builtin_include_args)

        candidates: List[Path] = []
        if self._configured_libclang:
            library_dir = self._configured_libclang.parent
            candidates.extend(
                [
                    library_dir.parent / "lib" / "clang",
                    library_dir.parent.parent / "lib" / "clang",
                    library_dir.parent / "clang",
                ]
            )

        builtin_include_args: List[str] = []
        seen: Set[str] = set()
        for candidate_root in candidates:
            if not candidate_root.exists():
                continue
            for version_dir in sorted(candidate_root.glob("*"), reverse=True):
                include_dir = version_dir / "include"
                normalized = self._normalize_path(include_dir)
                if include_dir.is_dir() and normalized not in seen:
                    builtin_include_args.extend(["-I", str(include_dir)])
                    seen.add(normalized)

        self._clang_builtin_include_args = builtin_include_args
        return list(self._clang_builtin_include_args)

    def _get_system_include_args(self) -> List[str]:
        if self._system_include_args is not None:
            return list(self._system_include_args)

        include_args: List[str] = []
        seen: Set[str] = set()

        def add_include_dir(path: Optional[Path]):
            if not path:
                return
            normalized = self._normalize_path(path)
            if not path.is_dir() or normalized in seen:
                return
            include_args.extend(["-I", str(path)])
            seen.add(normalized)

        vs_install = self._find_visual_studio_installation()
        if vs_install:
            msvc_root = vs_install / "VC" / "Tools" / "MSVC"
            if msvc_root.is_dir():
                versions = sorted([path for path in msvc_root.iterdir() if path.is_dir()])
                if versions:
                    add_include_dir(versions[-1] / "include")

        for sdk_root in self._candidate_windows_sdk_roots():
            if not sdk_root.is_dir():
                continue
            versions = sorted([path for path in sdk_root.iterdir() if path.is_dir()])
            if not versions:
                continue
            include_root = versions[-1]
            for subdir in ("ucrt", "um", "shared", "winrt", "cppwinrt"):
                add_include_dir(include_root / subdir)
            break

        self._system_include_args = include_args
        return list(self._system_include_args)

    def _get_forced_include_args(self) -> List[str]:
        if self._forced_include_args is not None:
            return list(self._forced_include_args)

        forced_args: List[str] = []

        vs_install = self._find_visual_studio_installation()
        msvc_include_dir: Optional[Path] = None
        if vs_install:
            msvc_root = vs_install / "VC" / "Tools" / "MSVC"
            if msvc_root.is_dir():
                versions = sorted(path for path in msvc_root.iterdir() if path.is_dir())
                if versions:
                    include_dir = versions[-1] / "include"
                    if include_dir.is_dir():
                        msvc_include_dir = include_dir

        if msvc_include_dir:
            # libclang misparses parts of the MSVC/UCRT/STL header lattice in
            # this repository unless these foundational headers are
            # established first.
            for header_name in ("vadefs.h", "stdint.h", "yvals_core.h"):
                header_path = msvc_include_dir / header_name
                if header_path.is_file():
                    forced_args.extend(["-include", str(header_path)])

        self._forced_include_args = forced_args
        return list(self._forced_include_args)

    def _find_visual_studio_installation(self) -> Optional[Path]:
        env_install = os.environ.get("VSINSTALLDIR")
        if env_install:
            install_path = Path(env_install)
            if install_path.exists():
                return install_path

        vswhere_candidates = [
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"),
            Path(r"C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe"),
        ]
        for vswhere in vswhere_candidates:
            if not vswhere.exists():
                continue
            try:
                import subprocess

                output = subprocess.check_output(
                    [
                        str(vswhere),
                        "-latest",
                        "-products",
                        "*",
                        "-requires",
                        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                        "-property",
                        "installationPath",
                    ],
                    encoding="utf-8",
                    errors="ignore",
                ).strip()
                if output:
                    return Path(output)
            except Exception:
                continue

        return None

    def _candidate_windows_sdk_roots(self) -> List[Path]:
        candidates: List[Path] = []

        env_sdk_dir = os.environ.get("WindowsSdkDir")
        if env_sdk_dir:
            candidates.append(Path(env_sdk_dir) / "Include")

        candidates.extend(
            [
                Path(r"C:\Program Files (x86)\Windows Kits\10\Include"),
            ]
        )
        return candidates

    def _find_vcxproj_for_file(self, file_path: Path) -> Optional[Path]:
        if not self.project_root or not self.build_root or not self.build_root.exists():
            return None

        try:
            relative_path = file_path.resolve().relative_to(self.project_root)
        except ValueError:
            return None

        parts = relative_path.parts
        if len(parts) < 4 or parts[:4] != ("external", "vulkancts", "modules", "vulkan"):
            return None

        if len(parts) == 5:
            candidate_dir = self.build_root / "external" / "vulkancts" / "modules" / "vulkan"
            for name in ("deqp-vk.vcxproj", "deqp-vksc.vcxproj"):
                candidate = candidate_dir / name
                if candidate.exists():
                    return candidate
            return None

        module_name = parts[4]
        candidate_dir = self.build_root / "external" / "vulkancts" / "modules" / "vulkan" / module_name
        if not candidate_dir.is_dir():
            return None

        preferred = [
            path
            for path in candidate_dir.glob("*.vcxproj")
            if path.name.startswith("deqp-vk-") and not path.name.endswith("-util.vcxproj")
        ]
        if preferred:
            return sorted(preferred)[0]

        generic = sorted(candidate_dir.glob("*.vcxproj"))
        return generic[0] if generic else None

    def _load_vcxproj_settings(self, vcxproj_path: Path) -> Dict:
        cache_key = self._normalize_path(vcxproj_path)
        if cache_key in self._vcxproj_cache:
            return self._vcxproj_cache[cache_key]

        tree = ET.parse(vcxproj_path)
        root = tree.getroot()

        cl_compile_node = None
        for config in self.preferred_configurations:
            condition = f"'$(Configuration)|$(Platform)'=='{config}'"
            cl_compile_node = root.find(
                f".//msb:ItemDefinitionGroup[@Condition=\"{condition}\"]/msb:ClCompile",
                self._MSBUILD_NS,
            )
            if cl_compile_node is not None:
                break

        if cl_compile_node is None:
            settings = {"include_args": [], "define_args": []}
            self._vcxproj_cache[cache_key] = settings
            return settings

        include_args: List[str] = []
        define_args: List[str] = []
        seen_includes: Set[str] = set()

        include_text = cl_compile_node.findtext(
            "msb:AdditionalIncludeDirectories",
            default="",
            namespaces=self._MSBUILD_NS,
        )
        define_text = cl_compile_node.findtext(
            "msb:PreprocessorDefinitions",
            default="",
            namespaces=self._MSBUILD_NS,
        )

        for entry in include_text.split(";"):
            entry = entry.strip()
            if not entry or entry.startswith("%("):
                continue

            expanded = self._expand_vcxproj_path(entry, vcxproj_path.parent)
            if not expanded or not expanded.is_dir():
                continue

            normalized = self._normalize_path(expanded)
            if normalized in seen_includes:
                continue

            include_args.extend(["-I", str(expanded)])
            seen_includes.add(normalized)

        for entry in define_text.split(";"):
            entry = entry.strip()
            if not entry or entry.startswith("%(") or "$(" in entry:
                continue
            if "__pragma" in entry:
                continue
            define_args.append("-D" + entry)

        settings = {
            "include_args": include_args,
            "define_args": define_args,
        }
        self._vcxproj_cache[cache_key] = settings
        return settings

    def _expand_vcxproj_path(self, raw_path: str, project_dir: Path) -> Optional[Path]:
        if "$(" in raw_path:
            if raw_path.startswith("$(ProjectDir)"):
                raw_path = raw_path.replace("$(ProjectDir)", str(project_dir) + os.sep)
            else:
                return None

        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (project_dir / candidate).resolve()
        return candidate

    def _get_default_project_include_args(self) -> List[str]:
        if not self.project_root:
            return []

        # 关键路径：保留原 20 个硬编码作为 v3.9 兼容性保险
        legacy_dirs = [
            self.project_root / "framework" / "common",
            self.project_root / "framework" / "delibs" / "debase",
            self.project_root / "framework" / "delibs" / "decpp",
            self.project_root / "framework" / "delibs" / "depool",
            self.project_root / "framework" / "delibs" / "dethread",
            self.project_root / "framework" / "delibs" / "deutil",
            self.project_root / "framework" / "delibs" / "destream",
            self.project_root / "framework" / "qphelper",
            self.project_root / "framework" / "referencerenderer",
            self.project_root / "external" / "vulkancts" / "framework" / "vulkan",
            self.project_root / "external" / "vulkancts" / "modules" / "vulkan",
            self.project_root / "external" / "vulkancts" / "modules" / "vulkan" / "util",
            self.project_root / "external" / "vulkancts" / "modules",
            self.project_root / "external" / "spirv-headers" / "src" / "include",
            self.project_root / "external" / "spirv-tools" / "src" / "include",
            self.project_root / "external" / "spirv-tools" / "src" / "external" / "include",
            self.project_root / "external" / "glslang" / "src",
            self.project_root / "external" / "glslang" / "src" / "External",
            self.project_root / "build" / "include",
            self.project_root / "build" / "external" / "vulkancts" / "framework" / "vulkan",
        ]

        # v3.10 修复：自动枚举 framework/ 下所有子目录（解决 addFunctionCase 函数引用解析失败）
        auto_dirs: List[Path] = []
        auto_roots = [
            self.project_root / "framework",
            self.project_root / "external" / "vulkancts" / "framework",
            self.project_root / "external" / "openglcts" / "framework",
            self.project_root / "external" / "vulkancts" / "common",
            self.project_root / "external" / "vulkancts" / "modules",
            self.project_root / "external" / "openglcts" / "common",
            self.project_root / "external" / "openglcts" / "modules",
        ]
        for top in auto_roots:
            if top.exists():
                for p in top.rglob("*"):
                    if p.is_dir():
                        auto_dirs.append(p)

        all_dirs = legacy_dirs + auto_dirs
        args: List[str] = []
        seen: Set[str] = set()
        for include_dir in all_dirs:
            normalized = self._normalize_path(include_dir)
            if not include_dir.is_dir() or normalized in seen:
                continue
            args.extend(["-I", str(include_dir)])
            seen.add(normalized)
        return args

    def _get_default_define_args(self) -> List[str]:
        return [
            "-DWIN32",
            "-D_WINDOWS",
            "-D_CRT_SECURE_NO_WARNINGS",
            "-D_SCL_SECURE_NO_WARNINGS",
            "-DDE_OS=DE_OS_WIN32",
            "-DDE_COMPILER=DE_COMPILER_MSC",
            "-DDE_CPU=DE_CPU_X86_64",
            "-DDE_PTR_SIZE=8",
            "-DDE_MINGW=0",
            "-DDE_ASSERT_FAILURE_CALLBACK",
            "-DDEQP_TARGET_NAME=\"Default\"",
            "-DDEQP_SUPPORT_DRM=0",
            "-DCTS_USES_VULKAN",
        ]

    def _cursor_text(self, file_path: Path, cursor) -> str:
        content = self._get_source_text(file_path)
        if not content:
            return ""

        lines = content.splitlines(keepends=True)
        start = cursor.extent.start
        end = cursor.extent.end

        if start.line < 1 or end.line < 1 or start.line > len(lines) or end.line > len(lines):
            return ""

        if start.line == end.line:
            line = lines[start.line - 1]
            return line[start.column - 1 : max(start.column - 1, end.column - 1)]

        segments = [lines[start.line - 1][start.column - 1 :]]
        for line_index in range(start.line, end.line - 1):
            segments.append(lines[line_index])
        segments.append(lines[end.line - 1][: max(0, end.column - 1)])
        return "".join(segments)

    def _get_source_text(self, file_path: Path) -> str:
        cache_key = self._normalize_path(file_path)
        if cache_key not in self._source_cache:
            self._source_cache[cache_key] = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        return self._source_cache[cache_key]

    def _extract_call_name(self, call_cursor) -> Optional[str]:
        callee_name = self._get_call_callee_name(call_cursor)
        if callee_name in (
            self._MODULE_CALL_NAMES
            | self._FUNCTION_CALL_NAMES
            | self._FUNCTION_WRAPPER_NAMES
            | self._FUNCTION_FACTORY_NAMES
            | self._CLASS_CALL_NAMES
        ):
            return callee_name
        return None

    def _get_call_callee_name(self, call_cursor) -> Optional[str]:
        try:
            tokens = [token.spelling for token in call_cursor.get_tokens()]
        except Exception:
            tokens = []

        if tokens:
            prefix = []
            for token in tokens:
                if token == '(':
                    break
                prefix.append(token)
            for token in reversed(prefix):
                if token.isidentifier() or '::' in token:
                    return token.split('::')[-1]

        referenced = getattr(call_cursor, "referenced", None)
        if referenced and getattr(referenced, "spelling", None):
            return referenced.spelling

        for child in call_cursor.get_children():
            child_ref = getattr(child, "referenced", None)
            if child_ref and getattr(child_ref, "spelling", None):
                return child_ref.spelling
            if child.spelling:
                return child.spelling
        return None

    def _string_literal_value(self, cursor, file_path: Path) -> str:
        text = self._cursor_text(file_path, cursor).strip()
        if text.startswith('"') and text.endswith('"') and len(text) >= 2:
            return text[1:-1]
        return text.strip('"')

    def _get_qualified_name(self, cursor) -> str:
        parts = [cursor.spelling]
        parent = cursor.semantic_parent
        while parent and parent.kind != clang.cindex.CursorKind.TRANSLATION_UNIT:
            if parent.spelling and parent.kind in {
                clang.cindex.CursorKind.NAMESPACE,
                clang.cindex.CursorKind.CLASS_DECL,
                clang.cindex.CursorKind.STRUCT_DECL,
                clang.cindex.CursorKind.CLASS_TEMPLATE,
            }:
                parts.append(parent.spelling)
            parent = parent.semantic_parent
        return "::".join(reversed(parts))

    def _get_namespace(self, cursor) -> str:
        parts: List[str] = []
        parent = cursor.semantic_parent
        while parent and parent.kind != clang.cindex.CursorKind.TRANSLATION_UNIT:
            if parent.spelling and parent.kind == clang.cindex.CursorKind.NAMESPACE:
                parts.append(parent.spelling)
            parent = parent.semantic_parent
        return "::".join(reversed(parts))

    def _container_name(self, cursor) -> Optional[str]:
        if not cursor:
            return None
        if cursor.kind == clang.cindex.CursorKind.NAMESPACE:
            return self._get_qualified_name(cursor)
        if cursor.kind in {
            clang.cindex.CursorKind.CLASS_DECL,
            clang.cindex.CursorKind.STRUCT_DECL,
            clang.cindex.CursorKind.CLASS_TEMPLATE,
        }:
            return self._get_qualified_name(cursor)
        return cursor.spelling or None

    def _normalize_type_name(self, spelling: str) -> str:
        cleaned = spelling.strip()
        for prefix in ("class ", "struct ", "enum ", "const "):
            cleaned = cleaned.replace(prefix, "")
        cleaned = cleaned.replace("&", "").replace("*", "")
        cleaned = " ".join(token for token in cleaned.split() if token != "const")
        return cleaned.strip()

    def _iter_descendants(self, cursor) -> Iterable:
        for child in cursor.get_children():
            yield child
            yield from self._iter_descendants(child)

    def _is_same_file_cursor(self, cursor, file_path: Path) -> bool:
        location = getattr(cursor, "location", None)
        location_file = getattr(location, "file", None)
        if not location_file:
            return False
        return self._normalize_path(location_file.name) == self._normalize_path(file_path)

    def _normalize_path(self, path_like) -> str:
        return os.path.normcase(os.path.normpath(str(path_like)))
