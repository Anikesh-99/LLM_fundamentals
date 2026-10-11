from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func, update
from .model import engine, Testcase, Project

class TestcaseManager:
    def __init__(self, project_id):
        self.project_id = project_id
        self.session = sessionmaker(bind=engine, expire_on_commit=False)
    
    def get_testcases(self, version=0):
        if not version or version == -1:
            with self.session() as session:
                version = session.scalar(
                    select(func.max(Testcase.version)).where(Testcase.project_id == self.project_id)
                )
        statement = select(Testcase).where(Testcase.project_id == self.project_id, Testcase.version == version)
        testcases = []
        with self.session() as session:
            testcases = session.scalars(statement).all()
        return testcases

    def get_testcase_by_type(self, type):
        statement = select(Testcase).where(Testcase.project_id == self.project_id, Testcase.type == type)
        with self.session() as session:
            testcases = session.scalars(statement).all()
            return testcases
    
    def update_test_case(self, testcase_id, new_testcase: Testcase):
        statement = (update(Testcase)
                     .where(Testcase.id == testcase_id)
                     .values(type=new_testcase.type, query=new_testcase.query, version=new_testcase.version, golden_answer=new_testcase.golden_answer))
        with self.session() as session:
            session.execute(statement)
            session.commit()

    def insert_test_case(self, testcase: Testcase):
        with self.session() as session:
            session.add(testcase)
            session.commit()
    
    def insert_test_cases(self, testcases: list[Testcase]):
        with self.session() as session:
            session.add_all(testcases)
            session.commit()
    
    def _duplicate(self, testcase: Testcase, version: int):
        data = dict(testcase.__dict__)
        data.pop('_sa_instance_state', None)
        data.pop('id', None)
        data['version'] = version
        return Testcase(**data)

    def add_new_version(self, changed_testcases: list[Testcase]) -> bool:
        try:
            testcases = self.get_testcases()
            version = changed_testcases[0].version
            changed_ids = [testcase.id for testcase in changed_testcases]
            with self.session() as session:
                for testcase in testcases:
                    if testcase.id not in changed_ids:
                        changed_testcases.append(self._duplicate(testcase, version))
            self.insert_test_cases(changed_testcases)
            return True
        except Exception as e:
            return False

class ProjectManager:
    def __init__(self, project_id):
        self.project_id = project_id
        self.session = sessionmaker(bind=engine, expire_on_commit=False)

    def get_project_type(self) -> str:
        with self.session() as session:
            statement = select(Project).where(Project.id == self.project_id)
            project = session.scalars(statement).all()
        return project[0].type if project else ""