import json
import os
from dataclasses import Field
from typing import Any, Dict, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (AIMessage, BaseMessage, ChatMessage,
                                     FunctionMessage, HumanMessage,
                                     SystemMessage, ToolMessage)
from langchain_core.outputs import (ChatGeneration, ChatGenerationChunk,
                                    ChatResult)
from langchain_core.utils import (convert_to_secret_str, get_from_dict_or_env,
                                  get_pydantic_field_names)
from langchain_core.utils.utils import build_extra_kwargs
from pydantic import SecretStr, root_validator
from snowflake.snowpark import Session


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any] = {
        "content": message.content,
    }

    # populate role and additional message data
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict



class ChatSnowflakeCortex(BaseChatModel):
    """Snowflake Cortex based Chat model"""
    
    sp_session: Session = None
    """Snowpark session. It is assumed database, role, warehouse etc.. are set before invoking the LLM"""

    model: str = 'snowflake-arctic'
    '''The Snowflake cortex hosted LLM model name. Defaulted to :snowflake-arctic. Refer to doc for other options. '''

    cortex_function: str = 'complete'
    '''The cortex function to use, defaulted to complete. for other types refer to doc'''

    snowflake_username: str = Field(alias="username")
    """Automatically inferred from env var `SNOWFLAKE_USERNAME` if not provided."""
    snowflake_password: Optional[SecretStr] = Field(default=None, alias="password")
    """Automatically inferred from env var `SNOWFLAKE_PASSWORD` if not provided."""
    snowflake_account: str = Field(alias="account")
    """Automatically inferred from env var `SNOWFLAKE_ACCOUNT` if not provided."""
    snowflake_database: str = Field(alias="database")
    """Automatically inferred from env var `SNOWFLAKE_DATABASE` if not provided."""
    snowflake_schema: str = Field(alias="schema")
    """Automatically inferred from env var `SNOWFLAKE_SCHEMA` if not provided."""
    snowflake_warehouse: str = Field(alias="warehouse")
    """Automatically inferred from env var `SNOWFLAKE_WAREHOUSE` if not provided."""
    snowflake_role: str = Field(alias="role")
    """Automatically inferred from env var `SNOWFLAKE_ROLE` if not provided."""


    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        values["model_kwargs"] = build_extra_kwargs(
            extra, values, all_required_field_names
        )
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        print("validating environment")
        values["snowflake_username"] = get_from_dict_or_env(values, "snowflake_username", "SNOWFLAKE_USERNAME")
        values["snowflake_password"] = convert_to_secret_str(
            get_from_dict_or_env(values, "snowflake_password", "SNOWFLAKE_PASSWORD")
        )
        values["snowflake_account"] = get_from_dict_or_env(values, "snowflake_account", "SNOWFLAKE_ACCOUNT")
        values["snowflake_database"] = get_from_dict_or_env(values, "snowflake_database", "SNOWFLAKE_DATABASE")
        values["snowflake_schema"] = get_from_dict_or_env(values, "snowflake_schema", "SNOWFLAKE_SCHEMA")
        values["snowflake_warehouse"] = get_from_dict_or_env(values, "snowflake_warehouse", "SNOWFLAKE_WAREHOUSE")
        values["snowflake_role"] = get_from_dict_or_env(values, "snowflake_role", "SNOWFLAKE_ROLE")

        connection_params = {
            "account": values["snowflake_account"],
            "user": values["snowflake_username"],
            "password": values["snowflake_password"].get_secret_value(),
            "database": values["snowflake_database"],
            "schema": values["snowflake_schema"],
            "warehouse": values["snowflake_warehouse"],
            "role": values["snowflake_role"],
        }
        
        try:
            values["sp_session"] = Session.builder.configs(connection_params).create()
            print("Session created successfully")
        except Exception as e:
            print("Failed to create session:", e)
        
        return values

    def __del__(self):
        if self.sp_session is not None:
            self.sp_session.close()

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return f"snowflake-cortex-{self.model}"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        message_str = str(message_dicts)
        options_str = str({'temperature': 0.7,'max_tokens': 10})
        sql_stmt = f'''
            select snowflake.cortex.{self.cortex_function}(
                '{self.model}'
                ,{message_str},{options_str}) as llm_reponse;'''
        
        l_rows = self.sp_session.sql(sql_stmt).collect()
        response = json.loads(l_rows[0]['LLM_REPONSE'])
        ai_message_content = response['choices'][0]['messages']

        message = AIMessage(
            content=ai_message_content,
            response_metadata= response['usage'],
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
