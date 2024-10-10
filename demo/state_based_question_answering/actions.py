
from sherpa_ai.actions.base import BaseAction
from sherpa_ai.events import Event, EventType
from sherpa_ai.memory.belief import Belief

class UserHelp(BaseAction):
    """
    Ask the user for clarification on a question.
    """
    args: dict = {"question": "str"}

    def execute(self, question: str) -> str:
        clarification = input(question)

        return clarification


class Respond(BaseAction):
    """
    Respond to the user with a message.
    """
    args: dict = {"response": "str"}

    def execute(self, response: str) -> str:
        print(response)

        return "success"


class StartQuestion(BaseAction):
    """
    Waiting the user to ask a question.
    """
    belief: Belief
    args: dict = {}

    def execute(self) -> str:
        question = input()
        self.belief.set_current_task(Event(EventType.task, "user", question))

        return "success"

class GenerateClass(BaseAction):
    """
    Respond to the user with a message.
    """
    args: dict = {}
    llm: any
    def execute(self) -> str:
        description = self.belief.get("description")
        task_description = self.belief.get("task_description")
        prompt = description + "\n\n" + task_description + "\n\n" + "Generate enumeration classes, abstract classes and regular classes with attributes."

        result = self.llm.predict(prompt)
        print(result)
        return result
    
class GenerateRelationship(BaseAction):
    """
    Respond to the user with a message.
    """
    args: dict = {}
    llm: any
    def execute(self) -> str:
        description = self.belief.get("description")
        task_description = self.belief.get("task_description")
        prompt = description + "\n\n" + task_description + "\n\n" + "Generate relationships based on the problem description."
        
        result = self.llm.predict(prompt)
        print(result)

        # post processing
        
        return result