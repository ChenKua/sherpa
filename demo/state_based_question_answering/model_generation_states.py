from sherpa_ai.actions.base import BaseAction
from sherpa_ai.actions.belief_actions import RetrieveBelief, UpdateBelief
from sherpa_ai.memory.belief import Belief
from sherpa_ai.memory.state_machine import SherpaStateMachine
from transitions.extensions import GraphMachine, HierarchicalGraphMachine

from actions import Respond, StartQuestion, UserHelp, GenerateElement


def get_actions(belief: Belief) -> dict[str, BaseAction]:
    start_question = StartQuestion(
        name="start_question",
        usage="Start the question answering process",
        belief=belief,
    )

    clarify_question = UserHelp(
        name="clarify_question",
        usage="Ask questions to clarify the intention of user",
        belief=belief,
    )
    answer_question = Respond(
        name="answer_question",
        usage="Answer the user's question based on the current context",
        belief=belief,
    )
    generate_element = GenerateElement(
        name="generate_element",
        usage="Answer the user's question based on the current context",
        belief=belief,
    )

    update_belief = UpdateBelief(belief=belief)

    retrieve_belief = RetrieveBelief(belief=belief)

    actions = [
        start_question,
        clarify_question,
        answer_question,
        update_belief,
        retrieve_belief,
    ]

    return {action.name: action for action in actions}


def add_mg_sm(belief: Belief) -> Belief:
    # Hierarchical version of the state machine
    states = [
        "Start",
        {
            "name": "ClassIdentificationState",
            "children": [
                "NounIdentification",
                "ClassIdentification",
                "AttributeIdentification",
                "EnumerationIdentification",
                "AbstractClassIdentification",
            ],
            "initial": "NounIdentification",
        },
        {
            "name": "PlayerRolePatternIdentificationState",
            "children": [
                "PatternIdentification",
                "PatternSummarization",
                "PatternIntegration",
            ],
            "initial": "PatternIdentification",
        },
        {
            "name": "FeedbackGenerationState",
            "children": ["FeedbackGeneration", "FeedbackIntegration"],
            "initial": "FeedbackGeneration",
        },
        {"name": "RelationshipIdentificationState"},
    ]
    initial = "Start"

    transitions = [
        {
            "trigger": "start",
            "source": "Start",
            "dest": "ClassIdentificationState",
        },
        {
            "trigger": "identifyNouns",
            "source": "ClassIdentificationState_NounIdentification",
            "dest": "ClassIdentificationState_ClassIdentification",
            "before":"", # things to do, different name with trigger
        },
        {
            "trigger": "identifyClasses",
            "source": "ClassIdentificationState_ClassIdentification",
            "dest": "ClassIdentificationState_AttributeIdentification",
        },
        {
            "trigger": "identifyAttributes",
            "source": "ClassIdentificationState_AttributeIdentification",
            "dest": "ClassIdentificationState_EnumerationIdentification",
        },
        {
            "trigger": "identifyEnumerations",
            "source": "ClassIdentificationState_EnumerationIdentification",
            "dest": "ClassIdentificationState_AbstractClassIdentification",
        },
        {
            "trigger": "identifyAbstractClass",
            "source": "ClassIdentificationState_AbstractClassIdentification",
            "dest": "PlayerRolePatternIdentificationState",
            # "before": "retrieve_belief",
        },
        {
            "trigger": "identifyPattern",
            "source": "PlayerRolePatternIdentificationState_PatternIdentification",
            "dest": "PlayerRolePatternIdentificationState_PatternSummarization",
        },
        # ???
        {
            "trigger": "summarizePattern",
            "source": "PlayerRolePatternIdentificationState_PatternSummarization",
            "dest": "PlayerRolePatternIdentificationState_PatternIntegration",
        },
        {
            "trigger": "integratePattern",
            "source": "PlayerRolePatternIdentificationState_PatternIntegration",
            "dest": "FeedbackGenerationState",
            "before": "update_belief",
        },
        {
            "trigger": "generateFeedback",
            "source": "FeedbackGenerationState_FeedbackGeneration",
            "dest": "FeedbackGenerationState_FeedbackIntegration",
        },
        {
            "trigger": "integrateFeedback",
            "source": "FeedbackGenerationState_FeedbackIntegration",
            "dest": "RelationshipIdentificationState",
        }
    ]

    action_map = get_actions(belief)

    sm = SherpaStateMachine(
        states=states,
        transitions=transitions,
        initial=initial,
        action_map=action_map,
        sm_cls=HierarchicalGraphMachine,
    )

    print(sm.sm.get_graph().draw(None))

    belief.state_machine = sm

    return belief
