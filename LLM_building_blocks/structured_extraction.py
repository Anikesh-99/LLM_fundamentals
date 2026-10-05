from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field
import os
from anthropic import Anthropic
import ValidationError

class Product(BaseModel):
    name: str
    price: float

class User(BaseModel):
    id: int
    email: EmailStr
    username: Optional[str] = None
    items: List[Product] = Field(default_factory=list)

user_schema = User.model_json_schema()

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

retries = 0
MAX_RETRY = 3
messages = [{"role": "user", "content": "Give me an example user to talk about"}]
while retries < MAX_RETRY:
    retries += 1
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet",
            max_tokens = 1024,
            messages = messages,
            tools = [{
                "name": "print_user_info",
                "description": "Prints user info",
                "input_schema": user_schema
            }],
            tool_choice={"type": "tool", "name": "print_user_info"}
        )
        tool_block = next(b for b in response.content if b.type == "tool_use")
        tool_input = response.content[0].input
        user = User(**tool_input)
        break
    except ValidationError as e:
        messages.append({"role": "assistant", "content": response.content})   # the bad tool_use
        messages.append({"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": tool_block.id,
            "is_error": True,
            "content": f"Validation failed. Fix these and call the tool again:\n{e}",
        }]})