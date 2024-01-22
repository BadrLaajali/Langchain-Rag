##To manage .env file
from dotenv import find_dotenv, load_dotenv

# Importing modules
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Charger les variables
load_dotenv(find_dotenv())


def generate_story(scenario):
    template = """
    You are a story teller;
    You can generate a short story based on a simple narrative, the story should be ne more than 20 words;
    CONTEXT :{scenario}
    STORY : 
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(
        llm=OpenAI(model_name="gpt-3.5-turbo", temperature=2),
        prompt=prompt,
        verbose=True,
    )

    story = story_llm.predict(scenario=scenario)

    print(story)
    return story


story = generate_story("thinking")
