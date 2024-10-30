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
        prompt = (
            description
            + "\n\n"
            + task_description
            + "\n\n"
            + "Generate enumeration classes, abstract classes and regular classes with attributes."
        )

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
        prompt = (
            description
            + "\n\n"
            + task_description
            + "\n\n"
            + "Generate relationships based on the problem description."
        )

        result = self.llm.predict(prompt)
        print(result)

        # post processing

        return result


class IdentifyNouns(BaseAction):
    """
    Identify nouns from text.
    """

    args: dict = {}
    llm: any

    def execute(self) -> str:
        description = self.belief.get("description")
        task_description = self.belief.get("task_description")
        noun_analysis_prompt = """
        Identify all the nouns in the description which can potentially be the class name, attribute name, role name.
        Include as much as nouns as possible and do not care about their functions for now.
        """
        format_description = """
        only output nouns and separated by , do not include any other words or symbels in your generated text.
        """
        prompt = (
            task_description
            + "\n\n"
            + description
            + "\n\n"
            + noun_analysis_prompt
            + "\n\n"
            + format_description
        )

        generated_text = self.llm.predict(prompt)

        noun_list = generated_text.split(",")
        noun_list = [
            i.strip() for i in noun_list if (i != "" and i != "\n" and i != None)
        ]
        print(noun_list)
        return noun_list


class IdentifyClasses(BaseAction):
    """
    Identify classes from the nouns list extracted from the problem description above.
    """

    args: dict = {}
    llm: any

    def execute(self) -> str:
        description = self.belief.get("description")
        task_description = self.belief.get("task_description")
        identify_classes_prompt = """
        Identify classes from the nouns list extracted from the problem description above.
        A class is the description for a set of similar objects that have the same structure and behavior, i.e., its instances
        All objects with the same features and behavior are instances of one class.
        In general, something should be a class if it could have instances.
        In general, something should be an instance if it is clearly a single member of the set defined by a class.
        Keep in mind that some of the nouns may be attributes or roles of the identified classes.
        Choose proper names for classes according the the following rules:
        1. Noun
        2. Singular
        3. Not too general, not too specific – at the right level of abstraction
        4. Avoid software engineering terms (data, record, table, information)
        5. Conventions: first letter capitalized; camel case without spaces if needed

        Example class names:
        Hospital, Doctor, PartTimeEmployee

        Constraints:
        Create classes at the right level of abstraction.
        Not all nouns in the nouns list are classes, some of them may be attributes, role names, or even not needed for diagram.
        Do NOT include all the nouns list as classes. Evaluate if it is needed to be a class.
        ONLY generate classes that are necessary to develop the system.


        Example:
        Problem Description: This system helps the Java Valley police officers keep track of the cases they are assigned to do. Officers may be assigned to investigate particular crimes, which involves interviewing victims at their homes and entering notes in the PI system.
        Identified Class List: PISystem, PoliceStation, Case, PoliceOfficer, Victim, Crime, Note
        """
        format_description = """
        only output class names and separated by , do not include any other words or symbols in your generated text.
        """
        nouns_list = self.belief.get("identify_nouns")
        prompt = (
            task_description
            + "\n\n"
            + description
            + "\n\n"
            + f"Nouns list: {nouns_list}"
            + "\n\n"
            + identify_classes_prompt
            + "\n\n"
            + format_description
            + "\n\n"
            + f"Identified Class List: \n"
        )

        generated_text = self.llm.predict(prompt)

        class_list = generated_text.split(",")
        class_list = [
            i.strip() for i in class_list if (i != "" and i != "\n" and i != None)
        ]
        print(class_list)
        return class_list


class IdentifyAttributes(BaseAction):
    """
    Identify attributes for each class.
    """

    args: dict = {}
    llm: any

    def execute(self) -> str:
        description = self.belief.get("description")
        task_description = self.belief.get("task_description")
        identify_attribute_prompt = """
        Given the current identify class list and noun list for potential class, attributes, role names.
        Identify attributes for each class.
        An attribute is a simple piece of data with a name and primitative datatype: string, int, date, time, boolean, etc
        More complex data is NOT modeled as an attribute.
        Attribute exists only when the object of the class exists.
        Conventions: first letter lower case; camel case without spaces if needed

        Notes:
        For each class, evaluate if it can be represented by an attrbute inside another class. If so, remove the class and make it an attribute.
        Do not include the class if it is not necessary in the software system.
        """
        format_description = """
        Follow the format for each class with its attribute: ClassName(type attributeName1, type attributeName2)
        For example:
        Person(string name, string address)
        only output class with attribute in () and separated by each line. do not include any other words or symbels in your generated text.
        """
        constraint = """
        You can overwrite the current class list if some classes are not necessary or should be attributes instead.
        Only generate attributes for the current classes.
        """

        nouns_list = self.belief.get("identify_nouns")
        class_list = self.belief.get("identify_classes")
        prompt = (
            task_description
            + "\n\n"
            + description
            + "\n\n"
            + f"Class list: {class_list}"
            + "\n\n"
            + f"Nouns list: {nouns_list}"
            + "\n\n"
            + identify_attribute_prompt
            + "\n\n"
            + constraint
            + "\n\n"
            + format_description
        )

        generated_text = self.llm.predict(prompt)

        class_attribute_list = generated_text.split("\n")
        class_attribute_list = [
            i.strip()
            for i in class_attribute_list
            if (i != "" and i != "\n" and i != None)
        ]

        print(class_attribute_list)
        return class_attribute_list


class IdentifyEnumerationClasses(BaseAction):
    """
    Identify enumeration classes for each class.
    """

    args: dict = {}
    llm: any

    def execute(self) -> str:
        description = self.belief.get("description")
        task_description = self.belief.get("task_description")
        identify_classes_prompt = """
        Identify enumeration classes from the current class.
        An enumeration class specifies a predefined list of choices, known as literals.
        Use the keyword "enum" to represent the class is an enumeration class
        For each literal, it consists of mainly one word, without any type.
        Do not show association with an enumeration, indicate as type of attribute.
        Often, the enumeration is defined as a single class, but is referenced for each of the class that needs the enumeration.
        In this case, it is used as an attribute, with the lower case of class name as attribute name and class name as attribute type.

        for example:
        enum PatronType(Student, Adult, Senior)
        LibyaryPatron(PatronType patronType)
        """

        format_description = """
        Follow the format for each class with its attribute: ClassName(type attributeName1, type attributeName2)
        Follow the format for each enumeration class with its literal: enum ClassName(Literal1, Literal2)
        For example:
        Person(string name, string address)
        enum Cake(WeddingCake, BirthdayCake)

        only output class with attribute in () and separated by each line. do not include any other words or symbels in your generated text.
        """

        constraint = """
        Only add the keyword enum if the original class should be an enumeration class
        Output all classes, including enumeration class and normal class
        """
        nouns_list = self.belief.get("identify_nouns")
        class_list = self.belief.get("identify_classes")
        class_attribute_list = self.belief.get("identify_attributes")

        prompt = (
            task_description
            + "\n\n"
            + description
            + "\n\n"
            + f"Class list: {class_attribute_list}"
            + "\n\n"
            + f"Nouns list: {nouns_list}"
            + "\n\n"
            + identify_classes_prompt
            + "\n\n"
            + constraint
            + "\n\n"
            + format_description
        )

        iteration_list = []

        for i in range(5):
            generated_text = self.llm.predict(prompt)

            class_list = generated_text.split("\n")
            class_list = [
                i.strip() for i in class_list if (i != "" and i != "\n" and i != None)
            ]
            iteration_list.extend(class_list)
        # print(class_list)
        return iteration_list


class IdentifyAbstractClasses(BaseAction):
    """
    Identify abstract classes from the current class.
    """

    args: dict = {}
    llm: any

    def execute(self) -> str:
        description = self.belief.get("description")
        task_description = self.belief.get("task_description")
        identify_classes_prompt = """
        Identify abstract classes from the current class.
        Abstract classes cannot be instantiated, i.e. the object of such class cannot be created directly using the new keyword

        We can treat an abstract class as a superclass and extend it:
        Structure and behavior specified for a superclass also applies to the subclass
        Subclass inherits from superclass

        for example:
        abstract Cake(int price)
        BirthdayCake(int numberOfCandles)
        WeddingCake(int numberOfTiers)

        for example:
        abstract Account(int balance, date openedDate, int creditorOverdraftLimit)
        MortgageAccount(int collateralValue)
        SavingsAccount()
        checkingAccount(int highestCheckNumber)

        use the keyword "abstract" to represent the class is abstract
        """

        format_description = """
        Follow the format for each class with its attribute: ClassName(type attributeName1, type attributeName2)
        For example:
        Person(string name, string address)
        abstract Account(int amount)
        only output class with attribute in () and separated by each line. do not include any other words or symbels in your generated text.
        """

        constraint = """
        Only add the keyword abstract if the original class should be an abstract class
        You can adjust the attributes within the subclass if the super class already contain the attribute
        Output all classes, including abstract classes, normal classes, and enumeration class
        """

        nouns_list = self.belief.get("identify_nouns")
        class_list = self.belief.get("identify_classes")
        class_attribute_list = self.belief.get("identify_attributes")
        class_attribute_enum_list = self.belief.get("identify_enumeration_classes")
        prompt = (
            task_description
            + "\n\n"
            + description
            + "\n\n"
            + f"Class list: {class_attribute_enum_list}"
            + "\n\n"
            + f"Nouns list: {nouns_list}"
            + "\n\n"
            + identify_classes_prompt
            + "\n\n"
            + constraint
            + "\n\n"
            + format_description
        )

        generated_text = self.llm.predict(prompt)

        class_list = generated_text.split("\n")
        class_list = [
            i.strip() for i in class_list if (i != "" and i != "\n" and i != None)
        ]
        print("=" * 40)
        print("partial model - class only")
        print(class_list)
        print("=" * 40)
        return class_list


class IdentifyPlayerRolePattern(BaseAction):
    """
    Identify potential player role pattern
    """

    args: dict = {}
    llm: any

    def execute(self) -> str:
        description = self.belief.get("description")
        task_description = self.belief.get("task_description")

        identify_classes_prompt = """
        identify the Player-Role pattern within the current classes
        for the classes Student, FullTimeStudent, PartTimeStudent, with the normal super class and subclass relationship,
        an instance of the Student cannot switch from FullTimeStudent to PartTimeStudent, as the instance cannot change type.
        So we need the player role pattern as following:

        Student(string name, string id)
        abstract AttendanceRole()
        FullTimeStudent(int fullTimeCredit) inherit AttendenceRole()
        PartTimeStudent(int partTimeCredit) inherit AttendenceRole()

        Here are more examples:

        Example 1. Within the school system, the student has two roles, graduate student and undergraduate student.
        The student can be a undergrad student at some point, and then switch to the role of graduate student.
        The student class saved information shared by both roles and is associated to the LevelRole.
        Both GraduateStudent and UndergradStudent inherit from the LevelRole class.

        Student(string name)
        abstract LevelRole()
        GraduateStudent(float graduateGpa) inherit LevelRole()
        UndergradStudentfloat undergradGpa inherit LevelRole()

        Example 2. Within the company system, each person has two roles, employee and manager.
        The Person can be an employee at some point, and then switch to the manager role later.

        Person(string name, string email, string address)
        abstract PersonRole()
        Employee(string employeeID) inherit PersonRole()
        Manager(string title) inherit PersonRole()

        Example 3. Within the system, each user has two roles, administrator and player.
        The user can be an administrator at some point, and then switch to the player role, or each user can have two roles at the same time.

        User(string userEmail, string userId)
        abstract UserRole()
        Administrator(string adminName, string adminPassword) inherit UserRole()
        Player(string playAccountName) inherit UserRole()

        Example 4. Within the conference system, each user has three roles: author, program chair, and reviewer.
        The user can have 1-3 roles at the same time. For example, the user can publish a paper as the author, work as a program chair, and review other papers at the same time.

        User(string username, string password)
        abstract UserRole()
        AuthorRole(string authorId) inherit UserRole()
        ProgramChairRole(string programCategory) inherit UserRole()
        ReviewerRole(string averageRating) inherit UserRole()

        Example 5. Within the company system, each person has at most 2 roles: a client, and an employee.
        The person can switch from a client to an employee, or keep two roles at the same time.
        For the role of employee, there are two types, lawer and low clerk. Both roles inherit from the employee role.

        Person(string name, string email)
        abstract UserRole()
        Client(string slientId) inherit UserRole()
        abstract Employee(string employeeId) inherit UserRole()
        Lawyer(string layerCategory) inherit Employee()
        LawClerk(string level) inherit Employee()
        """

        constraint = """
        Only output the classes that are within the Player-Role pattern.
        Do NOT include other classes.
        You may add new classes only if they are part of the Player-Role pattern.
        If there isn't any Player-Role pattern, simply say "No Player-Role pattern identified"
        Only generate Player-Role pattern within the description. Do not repeat the example.
        Ony use the Player-Role pattern when necessary according to the description.
        """

        format_description = """
        Follow the format for each class with its attribute: ClassName(type attributeName1, type attributeName2)
        Use the keyword "abstract" to represent the abstract class
        Use the keyword "inherit" to represent the subclass inherit attributes and relations from the super class
        """

        partial_model = self.belief.get("identify_abstract_classes")
        if self.belief.get("complete_model"):
            partial_model = self.belief.get("complete_model")
        prompt = (
            task_description
            + "\n\n"
            + description
            + "\n\n"
            + f"Class and attribute list: {partial_model}"
            + "\n\n"
            + identify_classes_prompt
            + "\n\n"
            + constraint
            + "\n\n"
            + format_description
        )

        generated_text = self.llm.predict(prompt)

        class_list = generated_text.split("\n")
        class_list = [
            i.strip() for i in class_list if (i != "" and i != "\n" and i != None)
        ]

        return class_list


class SummarizePlayerRolePattern(BaseAction):
    """
    Summarize player role pattern
    """

    args: dict = {}
    llm: any

    def execute(self) -> str:
        description = self.belief.get("description")
        task_description = self.belief.get("task_description")

        summarize_prompt = """Identify the Player-Role pattern from the descriotion provided with reference to five result list.
        Output the mostly like Player-Role pattern according to 5 result you have.
        You do not need to included everything from the 5 result you have, only include the classes you think it is correct.
        Combine the 5 result you have and make the final solution that make sense to you.
        Do not output other classes that are not included in the Player-Role pattern.
        If there isn't any Player-Role pattern, simply say "No Player-Role pattern identified"
        """

        format_description = """
        Follow the format for each class with its attribute: ClassName(type attributeName1, type attributeName2)
        Use the keyword "abstract" to represent the abstract class
        Use the keyword "inherit" to represent the subclass inherit attributes and relations from the super class.
        for example:
        Person(string name, string email, string address)
        abstract PersonRole()
        Employee(string employeeID) inherit PersonRole()
        Manager(string title) inherit PersonRole()
        """
        iteration_list = partial_model = self.belief.get("identify_player_role_pattern")
        partial_model = partial_model = self.belief.get("identify_abstract_classes")
        prompt = (
            f"Task description: {task_description}"
            + "\n\n"
            + summarize_prompt
            + "\n\n"
            + f"Description: {description}"
            + "\n\n"
            + f"5 solution list {iteration_list}"
            + "\n\n"
            + f"Class and attribute list: {partial_model}"
            + "\n\n"
            + f"Format description: {format_description}"
        )

        generated_text = self.llm.predict(prompt)

        return generated_text


class IntegrateClasses(BaseAction):
    """
    Integrate classes with player role pattern
    """

    args: dict = {}
    llm: any

    def execute(self) -> str:
        description = self.belief.get("description")
        task_description = self.belief.get("task_description")

        checker_prompt = """Using the current generated classes and identified player role pattern,
        combine the two versions and generate the final version of classes.

        Do the following things:
        1. analysis the generated classes to see if they are needed.
        Some generated classes may not be the right level of abstraction.
        Drop the classes if there are not necessary to describe the system.
        2. evaluate the player-role pattern to see if they are necessary.
        Not all system need the player-role pattern.
        Since player-role pattern can be complex in implementation, only use it if it is necessary.
        if the abstract classes and their subclasses are necessary, do not use player-role pattern.
        3. Combine the two version and make a solution that is consistent with both versions.
        Do not have duplicate classes in the final solution
        """

        format_description = """
        Do not generate other phrases besides the classes.
        Do not generate number for the classes.
        Follow the format for each class with its attribute: ClassName(type attributeName1, type attributeName2)
        Use the keyword "abstract" to represent the abstract class
        Use the keyword "inherit" to represent the subclass inherit attributes and relations from the super class.
        for example:
        Person(string name, string email, string address)
        abstract PersonRole()
        Employee(string employeeID) inherit PersonRole()
        Manager(string title) inherit PersonRole()
        """

        iteration_list = self.belief.get("identify_player_role_pattern")
        partial_model = self.belief.get("identify_abstract_classes")
        player_role_pattern = self.belief.get("summarize_player_role_pattern")
        prompt = (
            f"Task description: {task_description}"
            + "\n\n"
            + checker_prompt
            + "\n\n"
            + f"Description: {description}"
            + "\n\n"
            + f"Generated classes list {partial_model}"
            + "\n\n"
            + f"Player-role pattern: {player_role_pattern}"
            + "\n\n"
            + f"Format description: {format_description}"
            + "\n\n"
            + f"Integrated classes with attributes: \n"
        )

        generated_text = self.llm.predict(prompt)

        class_list = generated_text.split("\n")
        class_list = [
            i.strip() for i in class_list if (i != "" and i != "\n" and i != None)
        ]

        print("=" * 40)
        print("integrated classes with player role pattern")
        print(class_list)
        print("=" * 40)

        return class_list


class GenerateFeedback(BaseAction):
    """
    Write feedback for each class with its attribute.
    """

    args: dict = {}
    llm: any

    def execute(self) -> str:
        description = self.belief.get("description")
        task_description = self.belief.get("task_description")

        checker_prompt = """
        Given the class list for the problem description, write comment for each class with its attribute.
        Evaluate if it is at the correct level of abstraction to be included in the software system.
        Many classes may not be needed and may not be necessary, example cases:
        - if class A is too detailed to be included in the system, consider removing it.
        - if class A does not contain any attributes or only contains 1 attribute, consider moving the attribute of class A to another class and removing class A
        - For the enumeration class, evaluate if it should be captured by an attribute and if its literals are necessary
        - For the subclasses, evaluate if they are necessary to be present in the system.

        You can write general comments and comments to each class, evaluate if the class is necessary. If not, provide a solution to change it.
        """

        iteration_list = self.belief.get("identify_player_role_pattern")
        partial_model = self.belief.get("identify_abstract_classes")
        player_role_pattern = self.belief.get("summarize_player_role_pattern")
        complete_model = self.belief.get("integrate_classes")
        if not self.belief.get("complete_model") is None:
            complete_model = self.belief.get("complete_model")           
            
        prompt = (
            f"Task description: {task_description}"
            + "\n\n"
            + f"Problem Description: {description}"
            + "\n\n"
            + f"Class list: {complete_model}"
            + "\n\n"
            + checker_prompt
            + "\n\n"
            + f"Generated comments: \n"
        )

        generated_text = self.llm.predict(prompt)

        comments = generated_text.split("\n")
        comments = [
            i.strip() for i in comments if (i != "" and i != "\n" and i != None)
        ]
        print("\033[91mNew comments\033[0m")
        print(comments)

        return comments


class IntegrateFeedback(BaseAction):
    """
    Integrate feedback for each class with its attribute.
    """

    args: dict = {}
    llm: any

    def execute(self) -> str:
        description = self.belief.get("description")
        task_description = self.belief.get("task_description")

        integrate_prompt = """
        integrate the feedback given by the checker to finish the class diagram according to the problem description.
        """

        format_description = """
        Do not generate other phrases besides the classes.
        Do not generate number for the classes.
        Follow the format for each class with its attribute: ClassName(type attributeName1, type attributeName2)
        For example:
        Person(string name, string address)
        only output class with attribute in () and separated by each line. do not include any other words or symbels in your generated text.
        """

        iteration_list = self.belief.get("identify_player_role_pattern")
        partial_model = self.belief.get("identify_abstract_classes")
        player_role_pattern = self.belief.get("summarize_player_role_pattern")
        complete_model = self.belief.get("integrate_classes")
        
        if not self.belief.get("complete_model") is None:
            complete_model = self.belief.get("complete_model")   
            
        comments = self.belief.get("generate_feedback")
        prompt = (
            f"Task description: {task_description}"
            + "\n\n"
            + f"Problem Description: {description}"
            + "\n\n"
            + f"Class list: {complete_model}"
            + "\n\n"
            + f"Feedback from checker: {comments}"
            + "\n\n"
            + integrate_prompt
            + "\n\n"
            + format_description
            + f"Revised class diagram: "
        )

        generated_text = self.llm.predict(prompt)

        class_attribute_list = generated_text.split("\n")
        class_attribute_list = [
            i for i in class_attribute_list if (i != "" and i != "\n" and i != None)
        ]

        print("=" * 40)
        print("\033[91mrevised model - classes only\033[0m")
        print("revised model - classes only")
        print(class_attribute_list)
        print("=" * 40)
    
        self.belief.set("complete_model", class_attribute_list)  
        return class_attribute_list


class IdentifyRelationships(BaseAction):
    args: dict = {}
    llm: any

    def execute(self) -> str:
        description = self.belief.get("description")
        task_description = self.belief.get("task_description")

        relationship_prompt = """
        Identify relationships between classes. There are three types of relationships:

        1. Composition with the keyword "contain"
        example format: mul1 Class1 contain mul2 Class2
        Class1 and Class2 are classes above. mul1 and mul2 are one of the following options[0..*, 1, 0..1, 1..*]
        there might be multiple compositions
        In a typical domain model, there is usually a "system class" that contain most of the classes within the system
        For example:
        1 SchoolSystem contain 0..* UserRole
        1 SchoolSystem contain 0..* User
        1 SchoolSystem contain 0..* Course
        1 SchoolSystem contain 0..* Registration
        1 SchoolSystem contain 0..* StudentProfile

        2. Inheritance with the keyword "inherit"
        example format: Class1 inherit Class2
        Class1 and Class2 are classes above. there might be multiple inheritance
        Consider the inheritance relationship within the Player-Role pattern
        For example:
        Student inherit PersonRole
        Professor inherit PersonRole

        3. Association with the keyword "associate"
        example format: mul1 Class1 associate mul2 Class2
        Class1 and Class2 are classes above. mul1 and mul2 are one of the following options[0..*, 1, 0..1, 1..*]
        there might be multiple associations
        For example:
        0..* Student associate 0..5 Registration
        1 Student associate 0..1 StudentProfile

        Note:
        1. Use the classes in the given generated classes list, generate the classes (with attributes and enumeration abstract keywords) and their relationships.
        2. Only add the system class if the existing class diagram misses the system class.
        3. Do NOT change existing classes or add other classes besides the system class.
        4. In most of the cases, there is only 1 relationship within the same two classes.
        """

        format_description = """
        Generate the complete class diagram according to the class list using the following format:

        Classes:


        Relatipnships:
        Composition:

        Inheritance:

        Association:


        Make sure the generated text can be processed by text.split("\n") and then [text.strip()] into a list of processed classes and relationships
        """

        # iteration_list = self.belief.get("identify_player_role_pattern")
        # partial_model = self.belief.get("identify_abstract_classes")
        player_role_pattern = self.belief.get("summarize_player_role_pattern")
        # complete_model = self.belief.get("integrate_classes")
        revised_model = self.belief.get("integrate_feedback")
        if not self.belief.get("complete_model") is None:
            revised_model = self.belief.get("complete_model")
            
        prompt = (
            relationship_prompt
            + "\n\n"
            + f"Problem Description: {description}"
            + "\n\n"
            + f"Generated classes list: {revised_model}"
            + "\n\n"
            + f"Player-role pattern: {player_role_pattern}"
            + "\n\n"
            + f"Format description: {format_description}"
        )

        generated_text = self.llm.predict(prompt)
        complete_class_diagram = generated_text.split("\n")
        complete_class_diagram = [
            i.strip()
            for i in complete_class_diagram
            if (i != "" and i != "\n" and i != None)
        ]

        print("#"*40)
        print("\033[91m final model - \033[0m")
        for e in complete_class_diagram:
            print(e)
        # print(complete_class_diagram)
        print("#"*40)
        
        # identify if there is issue
        self.belief.get("complete_model", complete_class_diagram)
        return complete_class_diagram
