# Benchmark Data Dictionary

`benchmark_data.json` is a JSON array containing 406 OpenGL–Vulkan CTS test pairs. This dictionary describes only the fields stored in that file.

## Top-Level Fields

| Field | Type | Required | Meaning |
| --- | --- | ---: | --- |
| `row_id` | integer | yes | Stable identifier for the benchmark pair. |
| `gl_subclass` | object | yes | OpenGL CTS test-case record. |
| `gl_base_classes` | array of objects | yes | Base-class records associated with `gl_subclass`; may be empty. |
| `vk_recommended` | object | yes | Matched Vulkan CTS test-case record. |

## `gl_subclass` Fields

| Field | Type | Required | Meaning |
| --- | --- | ---: | --- |
| `class_id` | string | yes | Stable identifier for the extracted OpenGL test record. |
| `name` | string | no | Short test name when recorded. |
| `qualified_name` | string | yes | Namespace-qualified class or function name. |
| `namespace` | string | yes | Recorded source namespace; the value may be empty. |
| `file_path` | string | yes | Repository-relative source-file path. |
| `test_path` | string | yes | Registered CTS test path. |
| `code` | string | yes | Extracted test implementation. |
| `registration_source_file` | string | yes | Repository-relative path of the file containing the registration logic. |
| `registration_code` | string | yes | Source excerpt that registers or constructs the test. |
| `execute_result` | string | yes | Recorded CTS execution outcome. |

## `gl_base_classes` Item Fields

| Field | Type | Required | Meaning |
| --- | --- | ---: | --- |
| `class_id` | string | yes | Stable identifier for the OpenGL base-class record. |
| `name` | string | yes | Short base-class name. |
| `relationship` | string | yes | Recorded relationship between the test and the base class. |
| `qualified_name` | string | yes | Namespace-qualified base-class name. |
| `namespace` | string | yes | Recorded source namespace; the value may be empty. |
| `file_path` | string | yes | Repository-relative path of the base-class source file. |
| `code` | string | yes | Extracted base-class definition. |

## `vk_recommended` Fields

| Field | Type | Required | Meaning |
| --- | --- | ---: | --- |
| `class_id` | string | yes | Stable identifier for the extracted Vulkan test record. |
| `qualified_name` | string | yes | Namespace-qualified class or function name. |
| `namespace` | string | yes | Recorded source namespace. |
| `file_path` | string | yes | Repository-relative source-file path. |
| `test_path` | string | yes | Registered CTS test path. |
| `code` | string | yes | Extracted test implementation. |
| `registration_source_file` | string | yes | Repository-relative path of the file containing the registration logic. |
| `registration_code` | string | yes | Source excerpt that registers or constructs the test. |
| `execute_result` | string | yes | Recorded CTS execution outcome. |
| `base_classes` | array of objects | yes | Base-class records associated with the Vulkan test; may be empty. |

## `vk_recommended.base_classes` Item Fields

| Field | Type | Required | Meaning |
| --- | --- | ---: | --- |
| `class_id` | string | yes | Stable identifier for the Vulkan base-class record. |
| `name` | string | yes | Short base-class name. |
| `relationship` | string | yes | Recorded relationship between the test and the base class. |
| `qualified_name` | string | yes | Namespace-qualified base-class name. |
| `namespace` | string | yes | Recorded source namespace. |
| `file_path` | string | yes | Repository-relative path of the base-class source file. |
| `code` | string | yes | Extracted base-class definition. |
