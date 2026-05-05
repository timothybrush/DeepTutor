import time

from deeptutor.learning.models import (
    DiagnosticResult,
    ErrorRecord,
    ErrorType,
    KnowledgePoint,
    KnowledgeType,
    LearningModule,
    LearningProgress,
    LearningStage,
    MasteryLevel,
    QuizAttempt,
    RepetitionState,
    RetryAttempt,
    ReviewTask,
)


# ── Enums ────────────────────────────────────────────────────────────────

class TestKnowledgeType:
    def test_values(self):
        assert KnowledgeType.MEMORY.value == "记忆型"
        assert KnowledgeType.CONCEPT.value == "概念型"
        assert KnowledgeType.PROCEDURE.value == "程序型"
        assert KnowledgeType.DESIGN.value == "设计型"

    def test_str_subclass(self):
        assert isinstance(KnowledgeType.MEMORY, str)


class TestErrorType:
    def test_values(self):
        assert ErrorType.KNOWLEDGE_STRUCTURAL.value == "知识结构性"
        assert ErrorType.UNDERSTANDING_DEVIATION.value == "理解偏差型"
        assert ErrorType.APPLICATION_ERROR.value == "应用错误"
        assert ErrorType.METACOGNITIVE.value == "元认知型"


class TestMasteryLevel:
    def test_values(self):
        assert MasteryLevel.LEVEL_1 == 1
        assert MasteryLevel.MASTERED == 5

    def test_int_subclass(self):
        assert isinstance(MasteryLevel.MASTERED, int)


class TestLearningStage:
    def test_values(self):
        assert LearningStage.DIAGNOSTIC_PHASE1.value == "diagnostic_phase1"
        assert LearningStage.PRETEST.value == "pretest"
        assert LearningStage.COMPLETED.value == "completed"


# ── Models ───────────────────────────────────────────────────────────────

class TestKnowledgePoint:
    def test_instantiation(self):
        kp = KnowledgePoint(id="kp1", name="Ohm's Law", type=KnowledgeType.CONCEPT, module_id="m1")
        assert kp.id == "kp1"
        assert kp.type == KnowledgeType.CONCEPT

    def test_extra_ignored(self):
        kp = KnowledgePoint(id="kp1", name="x", type=KnowledgeType.MEMORY, module_id="m1", unknown=99)
        assert not hasattr(kp, "unknown") or kp.model_extra == {}


class TestLearningModule:
    def test_defaults(self):
        mod = LearningModule(id="m1", name="Circuits", order=1)
        assert mod.pass_threshold == 0.7
        assert mod.knowledge_points == []

    def test_with_knowledge_points(self):
        kp = KnowledgePoint(id="kp1", name="R", type=KnowledgeType.MEMORY, module_id="m1")
        mod = LearningModule(id="m1", name="C", order=1, knowledge_points=[kp])
        assert len(mod.knowledge_points) == 1
        assert mod.knowledge_points[0].name == "R"


class TestDiagnosticResult:
    def test_defaults(self):
        dr = DiagnosticResult()
        assert dr.module_mastery == {}
        assert dr.total_questions == 0
        assert dr.phase2_results == {}


class TestQuizAttempt:
    def test_defaults(self):
        qa = QuizAttempt(question_id="q1", knowledge_point_id="kp1", is_correct=True)
        assert qa.module_id == ""
        assert qa.error_type is None
        assert qa.mastery_estimate == 0.0
        assert isinstance(qa.timestamp, float)

    def test_with_error_type(self):
        qa = QuizAttempt(question_id="q1", knowledge_point_id="kp1", is_correct=False, error_type=ErrorType.APPLICATION_ERROR)
        assert qa.error_type == ErrorType.APPLICATION_ERROR


class TestRetryAttempt:
    def test_instantiation(self):
        ra = RetryAttempt(timestamp=time.time(), is_correct=True, attempt_number=2)
        assert ra.attempt_number == 2


class TestErrorRecord:
    def test_defaults(self):
        er = ErrorRecord(id="e1", question_id="q1", knowledge_point_id="kp1", module_id="m1", error_type=ErrorType.METACOGNITIVE)
        assert er.status == "active"
        assert er.retry_history == []
        assert isinstance(er.created_at, float)


class TestRepetitionState:
    def test_defaults(self):
        rs = RepetitionState(next_review_at=time.time())
        assert rs.interval_index == 0
        assert rs.consecutive_correct == 0
        assert rs.consecutive_wrong == 0


class TestReviewTask:
    def test_instantiation(self):
        rs = RepetitionState(next_review_at=time.time())
        rt = ReviewTask(id="r1", knowledge_point_id="kp1", knowledge_type=KnowledgeType.MEMORY, due_at=time.time(), priority=1, state=rs)
        assert rt.priority == 1


class TestLearningProgress:
    def test_defaults(self):
        lp = LearningProgress(book_id="b1")
        assert lp.current_stage == LearningStage.DIAGNOSTIC_PHASE1
        assert lp.learning_mode == "mastery"
        assert lp.current_kp_index == 0
        assert lp.modules == []
        assert isinstance(lp.created_at, float)

    def test_extra_allowed(self):
        lp = LearningProgress(book_id="b1", custom_field="hello")
        assert lp.model_extra.get("custom_field") == "hello"


# ── Serialization roundtrip ─────────────────────────────────────────────

class TestSerializationRoundtrip:
    def test_learning_progress_roundtrip(self):
        lp = LearningProgress(book_id="b1")
        lp.mastery_levels["kp1"] = 0.8
        lp.knowledge_types["kp1"] = KnowledgeType.CONCEPT
        data = lp.model_dump(mode="json")
        lp2 = LearningProgress.model_validate(data)
        assert lp2.book_id == "b1"
        assert lp2.mastery_levels["kp1"] == 0.8
        assert lp2.knowledge_types["kp1"] == KnowledgeType.CONCEPT

    def test_error_record_roundtrip(self):
        er = ErrorRecord(id="e1", question_id="q1", knowledge_point_id="kp1", module_id="m1", error_type=ErrorType.APPLICATION_ERROR)
        data = er.model_dump(mode="json")
        er2 = ErrorRecord.model_validate(data)
        assert er2.error_type == ErrorType.APPLICATION_ERROR
        assert er2.status == "active"
