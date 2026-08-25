"""cts_extractor_incremental — 给现有 libclang 流水线注入 4 字段。

不修改 src/vulkan_extractor 和 src/opengl_extractor 任何文件（只读参考），
只在本目录写新模块，被 ast_rag_modify/src/ast_vulkan/run_vulkan_ast_extract.py
和 ast_rag_modify/src/ast_opengl/run_opengl_ast_extract.py 在 record 拼装阶段调用。

公开 API（injector.py）：
  inject_4_fields(class_record, source_file, ast_utils, side, scan_root=None)

内部：
  mount_code_extractor.py — 找注册点所在函数体（mount_code）
  cross_file_scan.py     — 同目录+父目录找跨文件注册
  registration_classifier.py — 4 大类标签
"""