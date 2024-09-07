"""
This project idea is inspired by the larger NVIDIA [Deep Learning Institute course](https://www.nvidia.com/en-us/training/) titled [Building RAG Agents with LLMs.](https://www.nvidia.com/en-sg/training/instructor-led-workshops/building-rag-agents-with-llms//)
"""

!pip install langchain langchain_nvidia_ai_endpoints

from langchain_nvidia_ai_endpoints import ChatNVIDIA

# 取得可用模型清單
models = ChatNVIDIA.get_available_models(list_none=False)

# 格式化輸出清單
for model in models:
    if "meta" in model.id:
        print(f"Model: {model.id}, Type: {model.model_type}")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field

from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

console = Console()
base_style = Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)

########## Definition of RExtract
def RExtract(pydantic_class, llm, prompt):
    '''
    Runnable Extraction module
    Returns a knowledge dictionary populated by slot-filling extraction
    '''
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    instruct_merge = RunnableAssign({'format_instructions' : lambda x: parser.get_format_instructions()})
    def preparse(string):
        if '{' not in string: string = '{' + string
        if '}' not in string: string = string + '}'
        string = (string
            .replace("\\_", "_")
            .replace("\n", " ")
            .replace("\]", "]")
            .replace("\[", "[")
        )
        # print(string)  ## Good for diagnostics
        return string
    return instruct_merge | prompt | llm | preparse | parser
################################################################################

class KnowledgeBase(BaseModel):
   personality: str = Field('Unknown', description="User's preference for Extension or Introversion" )       #外向（Extraversion, E）與內向（Introversion, I）
   information_processing: str = Field('unknown', description="User's preference for Sensing or Intuition")     #感知（Sensing, S）與直覺（Intuition, N）
   decision_making: str = Field('unknown', description="User's preference for Thinking or Feeling")         #思考（Thinking, T）與情感（Feeling, F）
   lifestyle: str = Field('unknown', description="User's preference for Judging or Perceiving")           #判斷（Judging, J）與知覺（Perceiving, P）
   summary: str = Field('unknown', description="Running summary of user's MBTI traits")               #匯總結果
   response: str = Field('unknown', description="Ideal response based on user's personality type")         #回饋


# 系統人物設定
parser_prompt = ChatPromptTemplate.from_template(
    "You are MBTI consultant in psychological clinic."
    "Here is your question list"
    "How do you usually spend a free weekend?"
    "When starting a new travel plan, what is most important to you?"
    "Tell me about a difficult decision you've made recently. How did you make your choice?"
    "When traveling, do you prefer to plan everything in advance or go with the flow?"
    "By asking, 'How do you usually spend a free weekend?' aims to identify whether the user prefers social activities (Extraversion) or solitude (Introversion)."
    "By asking, 'When starting a new travel plan, what is most important to you?', we can discern if the user focuses on concrete details (Sensing) or overarching concepts (Intuition)."
    "By asking, 'Tell me about a difficult decision you've made recently. How did you make your choice?', we gauge whether the user relies on logical analysis (Thinking) or emotional factors (Feeling)."
    "By asking, 'When traveling, do you prefer to plan everything in advance or go with the flow?' helps determine if the user prefers structure (Judging) or spontaneity (Perceiving)"
    "Please combine Extraversion (E) or Introversion (I), Sensing (S) or Intuition (N), Thinking (T) or Feeling (F), and Judging (J) or Perceiving (P) into a four-letter code and output it to the summary."
    "You are chatting with a user. The user just responded ('input'). Please update the knowledge base."
    "Record your response in the 'response' tag to continue the conversation."
    "Do not hallucinate any details, and make sure the knowledge base is not redundant."
    "Update the entries frequently to adapt to the conversation flow."
    "\n{format_instructions}"
    "\n\nOLD KNOWLEDGE BASE: {know_base}"
    "\n\nNEW MESSAGE: {input}"
    "\n\nNEW KNOWLEDGE BASE:"
)
### TO　DO - Please enter your NVIDIA API key.
api_key = "Please enter your NVIDIA API key"


## Switch to a more powerful base model
instruct_llm = ChatNVIDIA(model="meta/llama3-70b-instruct", api_key=api_key) | StrOutputParser()

extractor = RExtract(KnowledgeBase, instruct_llm, parser_prompt)
info_update = RunnableAssign({'know_base' : extractor})

# show question to user
print("How do you usually spend a free weekend?")
print("When starting a new travel plan, what is most important to you?")
print("Tell me about a difficult decision you've made recently. How did you make your choice?")
print("When traveling, do you prefer to plan everything in advance or go with the flow?")

## Initialize the knowledge base and see what you get
state = {'know_base' : KnowledgeBase()}

state['input'] = "I usually spend my weekends hiking; the most important thing is to ask my girlfriend where she wants to go; I bought a phone case I really wanted, and as soon as I saw the color and style I liked, I immediately purchased it; plan ahead." # 對話1
state = info_update.invoke(state)
pprint(state)

state['input'] = "I like spending weekends by myself, reading books or researching topics that interest me, as this allows me to focus and think about future plans. I usually start by outlining the strategy and overall direction of a project to ensure that every step aligns with my long-term goals, and then I proceed with detailed planning. Recently, I had to decide whether to focus on my current career or pursue new challenges. I used logical analysis to weigh the pros and cons of each option and made the decision that I believe is the most strategic. I tend to plan my trips in advance, ensuring that every step follows the plan to avoid surprises and better manage my time and activities."
# 對話2
state = info_update.invoke(state)
pprint(state)
