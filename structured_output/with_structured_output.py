from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Annotated, Literal

load_dotenv()
model = ChatAnthropic(model = 'claude-3-5-sonnet-20241022')

class Review(BaseModel):
    key_themes: list[str] = Field(description = " Write down all the key themes discussed in the review in a list")
    summary: str = Field(decription = "A brief summary of the review")
    sentiment: Literal['pos', 'neg'] = Field(description = "Return sentiment of the review either negative, postive or neutral")
    pros: Optional[list[str]] = Field(default = None, description= "Write down all the pros inside a  list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a  list")
    name: Optional[str] = Field(default = None, description= "Write the name of hte review")


structured_model = model.with_structured_output(Review)
result = structured_model.invoke("""I’ve been using the SuperSonic Blender 3000 for a month now, and it’s honestly 
                                changed my mornings. The blender is extremely powerful, handling frozen fruits and 
                                ice with zero effort. I love how easy it is to clean—the detachable blades are a
                                 game-changer! The only downside is that it’s a bit noisy, especially on the highest 
                                 setting. Overall, if you’re looking for a reliable and fast blender, this is a 
                                 fantastic choice.""")
print(result)