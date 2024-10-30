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
    
    identify_nouns = IdentifyNouns(
        name="identify_nouns",
        usage="Identify nouns based on the modeling problem",
        belief=belief,
        llm=llm_class,
    )

    identify_classes = IdentifyClasses(
        name="identify_classes",
        usage="Identify classes based on the modeling problem",
        belief=belief,
        llm=llm_class,
    )
    
    identify_attributes = IdentifyAttributes(
        name="identify_attributes",
        usage="Identify attributes based on the modeling problem and nouns",
        belief=belief,
        llm=llm_class,
    )

    
    identify_enumeration_classes = IdentifyEnumerationClasses(
        name="identify_enumeration_classes",
        usage="Identify enumeration classes based on the current modeling problem",
        belief=belief,
        llm=llm_class,
    )
    
    identify_abstract_classes = IdentifyAbstractClasses(
        name="identify_abstract_classes",
        usage="Identify abstract classes based on the current modeling problem",
        belief=belief,
        llm=llm_class,
    )
    
    identify_player_role_pattern = IdentifyPlayerRolePattern(
        name="identify_player_role_pattern",
        usage="Identify player role pattern based on the current modeling problem",
        belief=belief,
        llm=llm_class,
    )

    summarize_player_role_pattern = SummarizePlayerRolePattern(
        name="summarize_player_role_pattern",
        usage="Summarize player role pattern based on the current modeling problem",
        belief=belief,
        llm=llm_class,
    )
    
    integrate_classes = IntegrateClasses(
        name="integrate_classes",
        usage="Integrate player role pattern into classes based on the current modeling problem",
        belief=belief,
        llm=llm_class,
    )
    
    generate_feedback = GenerateFeedback(
        name="generate_feedback",
        usage="Generate feeback bsed on the current domain model",
        belief=belief,
        llm=llm_class,
    )
    
    integrate_feedback = IntegrateFeedback(
        name="integrate_feedback",
        usage="Integrate feeback into the current domain model",
        belief=belief,
        llm=llm_class,
    )
    
    identify_relationships = IdentifyRelationships(
        name="identify_relationships",
        usage="Identify relationships based on the current domain model",
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
        identify_nouns,
        identify_classes,
        identify_enumeration_classes,
        identify_attributes,
        identify_abstract_classes,
        identify_player_role_pattern,
        summarize_player_role_pattern,
        integrate_classes,
        generate_feedback,
        integrate_feedback,
        identify_relationships,
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
        {"name":"InspectCompleteModel"},
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
            "before": "identify_nouns",  
        },
        # {
        #     "trigger": "Identify_nouns_again",
        #     "source": "ClassIdentificationState_ClassIdentification",
        #     "dest": "ClassIdentificationState_NounIdentification",  
        # },
        {
            "trigger": "Identify_classes",
            "source": "ClassIdentificationState_ClassIdentification",
            "dest": "ClassIdentificationState_AttributeIdentification",
            "before": "identify_classes",
        },
        # {
        #     "trigger": "Identify_classes_again",
        #     "source": "ClassIdentificationState_AttributeIdentification",
        #     "dest": "ClassIdentificationState_ClassIdentification",
        #     "before": "identify_classes",
        # },
        {
            "trigger": "Identify_attributes",
            "source": "ClassIdentificationState_AttributeIdentification",
            "dest": "ClassIdentificationState_EnumerationIdentification",
            "before": "identify_attributes",
        },
        # {
        #     "trigger": "Identify_attributes_again",
        #     "source": "ClassIdentificationState_EnumerationIdentification",
        #     "dest": "ClassIdentificationState_AttributeIdentification",
        # },
        {
            "trigger": "Identify_enumerations",
            "source": "ClassIdentificationState_EnumerationIdentification",
            "dest": "ClassIdentificationState_AbstractClassIdentification",
            "before": "identify_enumeration_classes",
        },
        # {
        #     "trigger": "Identify_enumerations_again",
        #     "source": "ClassIdentificationState_AbstractClassIdentification",
        #     "dest": "ClassIdentificationState_EnumerationIdentification",
        # },
        {
            "trigger": "Identify_abstract_classes",
            "source": "ClassIdentificationState_AbstractClassIdentification",
            "dest": "PlayerRolePatternIdentificationState",
            "before": "identify_abstract_classes",
        },
        # {
        #     "trigger": "Identify_abstract_classes_again",
        #     "source": "PlayerRolePatternIdentificationState",
        #     "dest": "ClassIdentificationState_AbstractClassIdentification",
        # },
        {
            "trigger": "Identify_pattern",
            "source": "PlayerRolePatternIdentificationState_PatternIdentification",
            "dest": "PlayerRolePatternIdentificationState_PatternSummarization",
            "before":"identify_player_role_pattern",
        },
        {
            "trigger": "Summarize_pattern",
            "source": "PlayerRolePatternIdentificationState_PatternSummarization",
            "dest": "PlayerRolePatternIdentificationState_PatternIntegration",
            "before": "summarize_player_role_pattern",
        },
        # {
        #     "trigger": "Summarize_pattern_again",
        #     "source": "PlayerRolePatternIdentificationState_PatternIntegration",
        #     "dest": "PlayerRolePatternIdentificationState_PatternSummarization",
        # },
        {
            "trigger": "Integrate_pattern",
            "source": "PlayerRolePatternIdentificationState_PatternIntegration",
            "dest": "FeedbackGenerationState",
            "before": "integrate_classes",
        },
        # {
        #     "trigger": "Integrate_pattern_again",
        #     "source": "FeedbackGenerationState",
        #     "dest": "PlayerRolePatternIdentificationState_PatternIntegration",
        # },
        {
            "trigger": "Generate_feedback",
            "source": "FeedbackGenerationState_FeedbackGeneration",
            "dest": "FeedbackGenerationState_FeedbackIntegration",
            "before": "generate_feedback",
        },
        # {
        #     "trigger": "Generate_feedback_again",
        #     "source": "FeedbackGenerationState_FeedbackIntegration",
        #     "dest": "FeedbackGenerationState_FeedbackGeneration",
        # },
        {
            "trigger": "Integrate_feedback",
            "source": "FeedbackGenerationState_FeedbackIntegration",
            "dest": "InspectCompleteModel",
            "before": "integrate_feedback",
        },
        # {
        #     "trigger": "Integrate_feedback_again",
        #     "source": "RelationshipIdentificationState",
        #     "dest": "FeedbackGenerationState_FeedbackIntegration",
        # },
        {
            "trigger": "finish",
            "source": "InspectCompleteModel",
            "dest": "end",
        },
        {
            "trigger": "Generate_feedback",
            "source": "InspectCompleteModel",
            "dest": "FeedbackGenerationState_FeedbackGeneration",
        },
        {
            "trigger": "Regenerate_class",
            "source": "InspectCompleteModel",
            "dest": "ClassIdentificationState",
        },
        {
            "trigger": "Regenerate_pattern",
            "source": "InspectCompleteModel",
            "dest": "PlayerRolePatternIdentificationState",
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
