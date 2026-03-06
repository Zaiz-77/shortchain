"""
Tool 基类与 @tool 装饰器。

用法一：装饰器（推荐）
    @tool
    def get_weather(city: str, unit: str = "celsius") -> str:
        '''获取指定城市的天气。

        :param city: 城市名称
        :param unit: 温度单位，celsius 或 fahrenheit
        '''
        ...

用法二：继承 Tool
    class MyTool(Tool):
        name = "my_tool"
        description = "做某件事"

        class ArgsSchema(BaseModel):
            x: int

        def run(self, x: int) -> str:
            return str(x)
"""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo


# --------------------------------------------------------------------------- #
# 类型 → JSON Schema 映射
# --------------------------------------------------------------------------- #

_PY_TYPE_TO_JSON: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _py_type_to_json_type(t: Any) -> str:
    return _PY_TYPE_TO_JSON.get(t, "string")


# --------------------------------------------------------------------------- #
# Tool 基类
# --------------------------------------------------------------------------- #


class Tool(ABC):
    """所有工具的基类。子类需声明 name、description，并实现 run()。"""

    name: str
    description: str

    # 子类可声明 ArgsSchema: type[BaseModel] 来精确描述参数
    ArgsSchema: type[BaseModel] | None = None

    def openai_schema(self) -> dict[str, Any]:
        """生成 OpenAI function calling 所需的 schema 字典。"""
        if self.ArgsSchema is not None:
            parameters = self.ArgsSchema.model_json_schema()
            _strip_schema_titles(parameters)
        else:
            parameters = {"type": "object", "properties": {}, "required": []}

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """执行工具，接收关键字参数，返回字符串结果。"""

    def __call__(self, **kwargs: Any) -> str:
        return self.run(**kwargs)


# --------------------------------------------------------------------------- #
# FunctionTool —— 包装普通函数
# --------------------------------------------------------------------------- #


class FunctionTool(Tool):
    """将一个普通 Python 函数包装为 Tool。由 @tool 装饰器创建。"""

    def __init__(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ):
        self._func = func
        self.name = name or func.__name__
        self.description = description or (inspect.getdoc(func) or "").split("\n")[0]
        self.ArgsSchema = _build_schema_from_func(func)

    def run(self, **kwargs: Any) -> str:
        result = self._func(**kwargs)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)

    def __repr__(self) -> str:
        return f"FunctionTool(name={self.name!r})"


# --------------------------------------------------------------------------- #
# 从函数签名自动构建 Pydantic schema
# --------------------------------------------------------------------------- #


def _build_schema_from_func(func: Callable[..., Any]) -> type[BaseModel]:
    """根据函数签名的类型注解和 docstring 参数描述生成 Pydantic 模型。"""
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    param_descriptions = _parse_param_docs(inspect.getdoc(func) or "")

    fields: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        annotation = hints.get(param_name, str)
        description = param_descriptions.get(param_name, "")

        if param.default is inspect.Parameter.empty:
            # 必填参数
            fields[param_name] = (annotation, FieldInfo(description=description))
        else:
            # 有默认值
            fields[param_name] = (
                annotation,
                FieldInfo(default=param.default, description=description),
            )

    model: type[BaseModel] = create_model(
        f"{func.__name__}_args",
        **fields,
    )
    return model


def _strip_schema_titles(schema: dict) -> None:
    """递归清理 Pydantic 生成的多余字段，确保与官方示例格式一致。"""
    for key in ("title", "$defs", "additionalProperties"):
        schema.pop(key, None)
    for prop in schema.get("properties", {}).values():
        if isinstance(prop, dict):
            for key in ("title", "$defs", "additionalProperties"):
                prop.pop(key, None)
    for sub in schema.get("$defs", {}).values():
        if isinstance(sub, dict):
            _strip_schema_titles(sub)


def _parse_param_docs(docstring: str) -> dict[str, str]:
    """
    简单解析 docstring 中的 :param name: desc 格式，返回 {param_name: description}。
    同时支持 Google 风格（Args: 小节）。
    """
    result: dict[str, str] = {}
    lines = docstring.splitlines()

    # :param name: description 格式
    for line in lines:
        line = line.strip()
        if line.startswith(":param "):
            rest = line[len(":param ") :]
            if ":" in rest:
                pname, _, pdesc = rest.partition(":")
                result[pname.strip()] = pdesc.strip()

    # Google 风格：Args: 小节下的 "name: description"
    in_args = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args = True
            continue
        if in_args:
            if stripped == "" or (
                stripped.endswith(":") and not stripped.startswith(" ")
            ):
                in_args = False
                continue
            if ":" in stripped:
                pname, _, pdesc = stripped.partition(":")
                pname = pname.strip()
                if pname and " " not in pname:
                    result.setdefault(pname, pdesc.strip())

    return result


# --------------------------------------------------------------------------- #
# @tool 装饰器
# --------------------------------------------------------------------------- #


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
    """
    将普通函数转为 Tool，支持两种用法：

        @tool
        def my_func(...): ...

        @tool(name="custom_name", description="自定义描述")
        def my_func(...): ...
    """
    if func is not None:
        # 不带参数的直接调用：@tool
        return FunctionTool(func, name=name, description=description)

    # 带参数的调用：@tool(...)
    def decorator(f: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(f, name=name, description=description)

    return decorator
