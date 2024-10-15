from sherpa_ai.actions.base import BaseAction
from sherpa_ai.actions.belief_actions import RetrieveBelief, UpdateBelief
from sherpa_ai.memory.belief import Belief
from sherpa_ai.memory.state_machine import SherpaStateMachine
from sherpa_ai.models import SherpaChatOpenAI
from transitions.extensions import GraphMachine, HierarchicalGraphMachine

from actions import (
    GenerateFeedback,
    IdentifyAbstractClasses,
    IdentifyAttributes,
    IdentifyClasses,
    IdentifyEnumerationClasses,
    IdentifyNouns,
    IdentifyPlayerRolePattern,
    IdentifyRelationships,
    IntegrateClasses,
    IntegrateFeedback,
    Respond,
    StartQuestion,
    SummarizePlayerRolePattern,
    UserHelp,
)

llm_class = SherpaChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
llm_relation = SherpaChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)


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

    identify_nouns = IdentifyNouns(
        name="identify_nouns",
        usage="Identify nouns from text.",
        belief=belief,
        llm=llm_class,
    )

    generate_classes = GenerateClass(
        name="generate_classes",
        usage="Generate class elements based on the current modeling problem",
        belief=belief,
        llm=llm_class,
    )

    generate_relationships = GenerateRelationship(
        name="generate_relationships",
        usage="Generate relationsihp elements based on the current modeling problem",
        belief=belief,
        llm=llm_relation,
    )
    identify_classes = IdentifyClasses(
        name="identify_classes",
        usage="Identify classes based on the current modeling problem",
        belief=belief,
        llm=llm_relation,
    )

    update_belief = UpdateBelief(belief=belief)

    retrieve_belief = RetrieveBelief(belief=belief)

    actions = [
        start_question,
        clarify_question,
        answer_question,
        update_belief,
        retrieve_belief,
        generate_classes,
        generate_relationships,
        identify_nouns,
    ]

    return {action.name: action for action in actions}


problem_description = """The LabTracker software helps (i) doctors manage the requisition of tests and examinations for patients and (ii) patients book appointments for tests and examinations at a lab. For the remainder of this description, tests and examinations are used interchangeably.
For a requisition, a doctor must provide their numeric practitioner number and signature for verification as well as their full name, their address, and their phone number. The signature is a digital signature, i.e., an image of the actual signature of the doctor. Furthermore, the doctor indicates the date from which the requisition is valid. The requisition must also show the patient?? information including their alpha-numeric health number, first name and last name, date of birth, address, and phone number. A doctor cannot prescribe a test for themselves but can prescribe tests to someone else who is a doctor.
Several tests can be combined on one requisition but only if they belong to the same group of tests. For example, only blood tests can be combined on one requisition or only ultrasound examinations can be combined. It is not possible to have a blood test and an ultrasound examination on the same requisition. For each test, its duration is defined by the lab network, so that it is possible to schedule appointments accordingly. The duration of a test is the same at each lab. For some kinds of tests, it does not matter how many tests are performed. They take as long as a single test. For example, several blood tests can be performed on a blood sample, i.e., it takes as long to draw the blood sample for a single blood test as it does for several blood tests.
A doctor may also indicate that the tests on a requisition are to be repeated for a specified number of times and interval. The interval is either weekly, monthly, every half year, or yearly. All tests on a requisition are following the same repetition pattern.
The doctor and the patient can view the results of each test (either negative or positive) as well as the accompanying report.
A patient is required to make an appointment for some tests while others are walk-in only. For example, x-ray examinations require an appointment, but blood tests are walk-in only (i.e., it is not possible to make an appointment for a blood test). On the other hand, some tests only require a sample to be dropped off (e.g., a urine or stool sample).
To make an appointment for a requisition, a patient selects the desired lab based on the lab?? address and business hours. For requisitions with repeated tests, a patient is only allowed to make one appointment at a time. The confirmation for an appointment also shows a confirmation number, the date as well as start/end times, and the name of the lab as well as its registration number. It is possible to change or cancel an appointment at any time but doing so within 24 hours of the appointment incurs a change/cancellation fee. Each lab determines its own fee and business hours. All labs are open every day of the year and offer all tests. The business hours of a lab do not change from one week to the next. Each day a lab is open from the day?? start time to its end time, i.e., there are no breaks.
"""

task_description = """
You are a domain modeling expert and are assigned with the task of domain modeling creation.
You objective is to create a textual based domain modeling given the program description.
There are steps involved in the process. Follow the instruction for your current step.
"""


def add_mg_sm(belief: Belief) -> Belief:
    # Hierarchical version of the state machine
    belief.set("description", problem_description)
    belief.set("task_description", task_description)

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
        {"name": "end"},
    ]
    initial = "Start"

    transitions = [
        {
            "trigger": "start",
            "source": "Start",
            "dest": "ClassIdentificationState",
        },
        {
            "trigger": "Identify_nouns",
            "source": "ClassIdentificationState_NounIdentification",
            "dest": "ClassIdentificationState_ClassIdentification",
            "before": "identify_nouns",  # things to do, different name with trigger
        },
        {
            "trigger": "Identify_classes",
            "source": "ClassIdentificationState_ClassIdentification",
            "dest": "ClassIdentificationState_AttributeIdentification",
            "before": "identify_classes",
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
            # "before": "update_belief",
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
        },
        {
            "trigger": "identifyRelationships",
            "source": "RelationshipIdentificationState",
            "dest": "end",
        },
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
