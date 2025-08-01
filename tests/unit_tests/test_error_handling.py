"""
Unit tests for MARL error handling system.

Tests error types, error handler, error analyzer, and recovery strategies.
"""

# Standard Library
import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

# Third-Party Library
import pytest

# SynThesisAI Modules
from core.marl.error_handling.error_analyzer import ErrorAnalyzer
from core.marl.error_handling.error_handler import ErrorHandlerFactory, MARLErrorHandler
from core.marl.error_handling.error_types import (
    AgentError,
    CommunicationError,
    CoordinationError,
    ErrorPattern,
    ErrorStatistics,
    LearningError,
    MARLError,
)
from core.marl.error_handling.recovery_strategies import (
    AgentRestartStrategy,
    CommunicationRetryStrategy,
    CoordinationResetStrategy,
    FallbackStrategy,
    LearningResetStrategy,
    RecoveryResult,
    RecoveryStrategy,
    RecoveryStrategyManager,
)


class TestErrorTypes:
    """Test error type classes."""

    def test_marl_error_creation(self):
        """Test basic MARL error creation."""
        error = MARLError(
            "Test error message",
            error_code="TEST_ERROR",
            context={"test_key": "test_value"},
            recovery_hint="Test recovery hint",
            severity="ERROR",
        )

        assert error.message == "Test error message"
        assert error.error_code == "TEST_ERROR"
        assert error.context["test_key"] == "test_value"
        assert error.recovery_hint == "Test recovery hint"
        assert error.severity == "ERROR"
        assert error.error_id.startswith("TEST_ERROR_")
        assert isinstance(error.timestamp, datetime)

    def test_marl_error_context_management(self):
        """Test error context management."""
        error = MARLError("Test error")

        # Add context
        error.add_context("key1", "value1")
        error.add_context("key2", 42)

        assert error.get_context("key1") == "value1"
        assert error.get_context("key2") == 42
        assert error.get_context("nonexistent", "default") == "default"

    def test_marl_error_to_dict(self):
        """Test error serialization to dictionary."""
        error = MARLError("Test error", error_code="TEST_ERROR", context={"test": "value"})

        error_dict = error.to_dict()

        assert error_dict["error_code"] == "TEST_ERROR"
        assert error_dict["message"] == "Test error"
        assert error_dict["context"]["test"] == "value"
        assert "error_id" in error_dict
        assert "timestamp" in error_dict

    def test_agent_error_creation(self):
        """Test agent error creation."""
        error = AgentError("Agent failed", agent_id="test_agent", agent_type="generator")

        assert error.agent_id == "test_agent"
        assert error.agent_type == "generator"
        assert error.get_context("agent_id") == "test_agent"
        assert error.get_context("agent_type") == "generator"

    def test_coordination_error_creation(self):
        """Test coordination error creation."""
        error = CoordinationError(
            "Coordination failed",
            coordination_id="coord_123",
            participating_agents=["agent1", "agent2"],
        )

        assert error.coordination_id == "coord_123"
        assert error.participating_agents == ["agent1", "agent2"]
        assert error.get_context("coordination_id") == "coord_123"
        assert error.get_context("participating_agents") == ["agent1", "agent2"]

    def test_error_pattern_creation(self):
        """Test error pattern creation."""
        pattern = ErrorPattern(pattern_id="test_pattern", error_codes=["ERROR_1", "ERROR_2"])

        assert pattern.pattern_id == "test_pattern"
        assert pattern.error_codes == ["ERROR_1", "ERROR_2"]
        assert pattern.frequency == 0
        assert pattern.recovery_success_rate == 0.0

    def test_error_pattern_matching(self):
        """Test error pattern matching."""
        pattern = ErrorPattern(pattern_id="test_pattern", error_codes=["TEST_ERROR", "OTHER_ERROR"])

        matching_error = MARLError("Test", error_code="TEST_ERROR")
        non_matching_error = MARLError("Test", error_code="DIFFERENT_ERROR")

        assert pattern.matches(matching_error) is True
        assert pattern.matches(non_matching_error) is False

    def test_error_pattern_frequency_update(self):
        """Test error pattern frequency updates."""
        pattern = ErrorPattern(pattern_id="test_pattern", error_codes=["TEST_ERROR"])

        initial_time = pattern.last_occurrence
        pattern.update_frequency()

        assert pattern.frequency == 1
        assert pattern.last_occurrence != initial_time

    def test_error_statistics_recording(self):
        """Test error statistics recording."""
        stats = ErrorStatistics()

        error1 = AgentError("Agent error", "agent1")
        error2 = CoordinationError("Coordination error")

        stats.record_error(error1)
        stats.record_error(error2)

        assert stats.total_errors == 2
        assert stats.errors_by_type["AgentError"] == 1
        assert stats.errors_by_type["CoordinationError"] == 1
        assert stats.errors_by_severity["ERROR"] == 2

    def test_error_statistics_recovery_tracking(self):
        """Test recovery attempt tracking."""
        stats = ErrorStatistics()

        stats.record_recovery_attempt(True, 1.5)
        stats.record_recovery_attempt(False, 2.0)
        stats.record_recovery_attempt(True, 1.0)

        assert stats.recovery_attempts == 3
        assert stats.successful_recoveries == 2
        assert stats.get_recovery_success_rate() == 2 / 3
        assert stats.average_recovery_time == (1.5 + 2.0 + 1.0) / 3


class TestRecoveryStrategies:
    """Test recovery strategy classes."""

    def test_recovery_result_creation(self):
        """Test recovery result creation."""
        result = RecoveryResult(
            success=True,
            strategy_name="test_strategy",
            recovery_time=1.5,
            details={"action": "restart"},
        )

        assert result.success is True
        assert result.strategy_name == "test_strategy"
        assert result.recovery_time == 1.5
        assert result.details["action"] == "restart"

    def test_recovery_result_to_dict(self):
        """Test recovery result serialization."""
        result = RecoveryResult(
            success=False, strategy_name="test_strategy", error="Recovery failed"
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is False
        assert result_dict["strategy_name"] == "test_strategy"
        assert result_dict["error"] == "Recovery failed"

    @pytest.mark.asyncio
    async def test_agent_restart_strategy(self):
        """Test agent restart strategy."""
        strategy = AgentRestartStrategy()

        # Test can_handle
        agent_error = AgentError("Agent failed", "test_agent", error_code="AGENT_INIT_ERROR")
        other_error = CoordinationError("Coordination failed")

        assert await strategy.can_handle(agent_error) is True
        assert await strategy.can_handle(other_error) is False

        # Mock the restart methods
        with (
            patch.object(strategy, "_restart_agent", new_callable=AsyncMock) as mock_restart,
            patch.object(
                strategy,
                "_verify_agent_health",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_health,
        ):
            result = await strategy.execute_recovery(agent_error)

            assert result.success is True
            assert result.strategy_name == "agent_restart"
            assert result.details["agent_id"] == "test_agent"
            mock_restart.assert_called_once_with("test_agent")
            mock_health.assert_called_once_with("test_agent")

    @pytest.mark.asyncio
    async def test_coordination_reset_strategy(self):
        """Test coordination reset strategy."""
        strategy = CoordinationResetStrategy()

        # Test can_handle
        coord_error = CoordinationError("Coordination failed", coordination_id="coord_123")
        other_error = AgentError("Agent failed", "test_agent")

        assert await strategy.can_handle(coord_error) is True
        assert await strategy.can_handle(other_error) is False

        # Mock the reset methods
        with (
            patch.object(
                strategy, "_reset_coordination_state", new_callable=AsyncMock
            ) as mock_reset,
            patch.object(
                strategy,
                "_check_agent_health",
                new_callable=AsyncMock,
                return_value=["agent1", "agent2"],
            ) as mock_health,
        ):
            result = await strategy.execute_recovery(coord_error)

            assert result.success is True
            assert result.strategy_name == "coordination_reset"
            mock_reset.assert_called_once()
            mock_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_learning_reset_strategy(self):
        """Test learning reset strategy."""
        strategy = LearningResetStrategy()

        # Test can_handle
        learning_error = LearningError(
            "Learning divergence detected", learning_component="q_network"
        )
        other_error = LearningError("Buffer full", learning_component="buffer")

        assert await strategy.can_handle(learning_error) is True
        assert await strategy.can_handle(other_error) is False

        # Mock the reset methods
        with (
            patch.object(
                strategy, "_reset_learning_parameters", new_callable=AsyncMock
            ) as mock_params,
            patch.object(
                strategy, "_clear_experience_buffers", new_callable=AsyncMock
            ) as mock_buffers,
            patch.object(
                strategy, "_reinitialize_learning_state", new_callable=AsyncMock
            ) as mock_init,
        ):
            result = await strategy.execute_recovery(learning_error)

            assert result.success is True
            assert result.strategy_name == "learning_reset"
            mock_params.assert_called_once()
            mock_buffers.assert_called_once()
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_communication_retry_strategy(self):
        """Test communication retry strategy."""
        strategy = CommunicationRetryStrategy(max_retries=2)

        # Test can_handle
        comm_error = CommunicationError("Message failed", sender="agent1", receiver="agent2")
        other_error = AgentError("Agent failed", "test_agent")

        assert await strategy.can_handle(comm_error) is True
        assert await strategy.can_handle(other_error) is False

        # Mock successful retry
        with patch.object(
            strategy, "_retry_communication", new_callable=AsyncMock, return_value=True
        ) as mock_retry:
            result = await strategy.execute_recovery(comm_error)

            assert result.success is True
            assert result.strategy_name == "communication_retry"
            assert result.details["retry_attempt"] == 1
            mock_retry.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_strategy(self):
        """Test fallback strategy."""
        strategy = FallbackStrategy()

        # Test can_handle (should handle any error)
        any_error = MARLError("Any error")
        assert await strategy.can_handle(any_error) is True

        # Mock fallback methods
        with (
            patch.object(strategy, "_log_error_details", new_callable=AsyncMock) as mock_log,
            patch.object(strategy, "_notify_administrators", new_callable=AsyncMock) as mock_notify,
            patch.object(strategy, "_enable_safe_mode", new_callable=AsyncMock) as mock_safe,
        ):
            result = await strategy.execute_recovery(any_error)

            assert result.success is True
            assert result.strategy_name == "fallback"
            assert result.details["safe_mode_enabled"] is True
            mock_log.assert_called_once()
            mock_notify.assert_called_once()
            mock_safe.assert_called_once()

    def test_recovery_strategy_statistics(self):
        """Test recovery strategy statistics tracking."""
        strategy = AgentRestartStrategy()

        # Record some attempts
        strategy.record_attempt(True, 1.0)
        strategy.record_attempt(False, 2.0)
        strategy.record_attempt(True, 1.5)

        assert strategy.attempts == 3
        assert strategy.successes == 2
        assert strategy.get_success_rate() == 2 / 3
        assert strategy.get_average_recovery_time() == (1.0 + 2.0 + 1.5) / 3

        stats = strategy.get_statistics()
        assert stats["attempts"] == 3
        assert stats["successes"] == 2
        assert stats["success_rate"] == 2 / 3


class TestRecoveryStrategyManager:
    """Test recovery strategy manager."""

    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = RecoveryStrategyManager()

        assert len(manager.strategies) > 0

        # Check that strategies are sorted by priority
        priorities = [s.priority for s in manager.strategies]
        assert priorities == sorted(priorities, reverse=True)

    @pytest.mark.asyncio
    async def test_get_recovery_strategy(self):
        """Test getting recovery strategy for error."""
        manager = RecoveryStrategyManager()

        # Test agent error
        agent_error = AgentError("Agent failed", "test_agent", error_code="AGENT_INIT_ERROR")
        strategy = await manager.get_recovery_strategy(agent_error)

        assert strategy is not None
        assert strategy.strategy_name == "agent_restart"

        # Test coordination error
        coord_error = CoordinationError("Coordination failed")
        strategy = await manager.get_recovery_strategy(coord_error)

        assert strategy is not None
        assert strategy.strategy_name == "coordination_reset"

    def test_add_remove_strategy(self):
        """Test adding and removing strategies."""
        manager = RecoveryStrategyManager()
        initial_count = len(manager.strategies)

        # Create custom strategy
        class CustomStrategy(RecoveryStrategy):
            def __init__(self):
                super().__init__("custom_strategy", priority=9)

            async def can_handle(self, error):
                return False

            async def execute_recovery(self, error):
                return RecoveryResult(True, self.strategy_name)

        custom_strategy = CustomStrategy()

        # Add strategy
        manager.add_strategy(custom_strategy)
        assert len(manager.strategies) == initial_count + 1

        # Check it's in the right position (high priority)
        assert manager.strategies[0].strategy_name == "custom_strategy"

        # Remove strategy
        success = manager.remove_strategy("custom_strategy")
        assert success is True
        assert len(manager.strategies) == initial_count

        # Try to remove non-existent strategy
        success = manager.remove_strategy("nonexistent")
        assert success is False

    def test_get_strategy_statistics(self):
        """Test getting strategy statistics."""
        manager = RecoveryStrategyManager()

        stats = manager.get_strategy_statistics()

        assert isinstance(stats, list)
        assert len(stats) == len(manager.strategies)

        for stat in stats:
            assert "strategy_name" in stat
            assert "priority" in stat
            assert "attempts" in stat
            assert "successes" in stat

    def test_get_strategy_by_name(self):
        """Test getting strategy by name."""
        manager = RecoveryStrategyManager()

        strategy = manager.get_strategy_by_name("agent_restart")
        assert strategy is not None
        assert strategy.strategy_name == "agent_restart"

        strategy = manager.get_strategy_by_name("nonexistent")
        assert strategy is None


class TestErrorAnalyzer:
    """Test error analyzer."""

    @pytest.fixture
    def temp_persistence_path(self):
        """Create temporary persistence path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "test_patterns.json"

    def test_analyzer_initialization(self, temp_persistence_path):
        """Test analyzer initialization."""
        analyzer = ErrorAnalyzer(
            pattern_window_size=50,
            pattern_threshold=2,
            persistence_path=temp_persistence_path,
        )

        assert analyzer.pattern_window_size == 50
        assert analyzer.pattern_threshold == 2
        assert analyzer.persistence_path == temp_persistence_path
        assert len(analyzer.recent_errors) == 0

    @pytest.mark.asyncio
    async def test_analyze_error(self, temp_persistence_path):
        """Test single error analysis."""
        analyzer = ErrorAnalyzer(pattern_threshold=2, persistence_path=temp_persistence_path)

        error = AgentError("Test error", "test_agent")
        result = await analyzer.analyze_error(error)

        assert "error_id" in result
        assert "matching_patterns" in result
        assert "recommendations" in result
        assert "severity_assessment" in result
        assert len(analyzer.recent_errors) == 1

    @pytest.mark.asyncio
    async def test_pattern_identification(self, temp_persistence_path):
        """Test pattern identification."""
        analyzer = ErrorAnalyzer(
            pattern_threshold=2,
            analysis_interval=0.1,
            persistence_path=temp_persistence_path,
        )

        # Add multiple similar errors
        for i in range(3):
            error = AgentError(f"Test error {i}", "test_agent", error_code="AGENT_TEST_ERROR")
            await analyzer.analyze_error(error)

        # Trigger pattern analysis
        await analyzer._perform_pattern_analysis()

        # Check that patterns were identified
        assert len(analyzer.error_patterns) > 0

    @pytest.mark.asyncio
    async def test_start_stop_analysis(self, temp_persistence_path):
        """Test starting and stopping analysis."""
        analyzer = ErrorAnalyzer(analysis_interval=0.1, persistence_path=temp_persistence_path)

        # Start analysis
        await analyzer.start_analysis()
        assert analyzer.is_running is True
        assert analyzer.analysis_task is not None

        # Stop analysis
        await analyzer.stop_analysis()
        assert analyzer.is_running is False

    def test_pattern_persistence(self, temp_persistence_path):
        """Test pattern persistence."""
        # Create analyzer and add some patterns
        analyzer = ErrorAnalyzer(enable_persistence=True, persistence_path=temp_persistence_path)

        # Add a pattern manually
        pattern = ErrorPattern(pattern_id="test_pattern", error_codes=["TEST_ERROR"], frequency=5)
        analyzer.error_patterns["test_pattern"] = pattern

        # Save patterns
        analyzer._save_patterns()

        # Verify file was created
        assert temp_persistence_path.exists()

        # Create new analyzer and load patterns
        new_analyzer = ErrorAnalyzer(
            enable_persistence=True, persistence_path=temp_persistence_path
        )

        # Verify pattern was loaded
        assert "test_pattern" in new_analyzer.error_patterns
        assert new_analyzer.error_patterns["test_pattern"].frequency == 5

    def test_get_pattern_summary(self, temp_persistence_path):
        """Test getting pattern summary."""
        analyzer = ErrorAnalyzer(pattern_threshold=2, persistence_path=temp_persistence_path)

        # Add some patterns
        pattern1 = ErrorPattern("pattern1", ["ERROR1"], frequency=5)
        pattern2 = ErrorPattern("pattern2", ["ERROR2"], frequency=3)
        analyzer.error_patterns["pattern1"] = pattern1
        analyzer.error_patterns["pattern2"] = pattern2

        summary = analyzer.get_pattern_summary()

        assert summary["total_patterns"] == 2
        assert summary["active_patterns"] == 2
        assert len(summary["most_frequent_patterns"]) == 2
        assert summary["most_frequent_patterns"][0][0] == "pattern1"  # Most frequent first

    def test_get_pattern_details(self, temp_persistence_path):
        """Test getting pattern details."""
        analyzer = ErrorAnalyzer(persistence_path=temp_persistence_path)

        pattern = ErrorPattern(
            "test_pattern", ["TEST_ERROR"], frequency=3, recovery_success_rate=0.8
        )
        analyzer.error_patterns["test_pattern"] = pattern

        details = analyzer.get_pattern_details("test_pattern")

        assert details is not None
        assert details["pattern_id"] == "test_pattern"
        assert details["frequency"] == 3
        assert details["recovery_success_rate"] == 0.8
        assert "recommendations" in details

        # Test non-existent pattern
        details = analyzer.get_pattern_details("nonexistent")
        assert details is None


class TestMARLErrorHandler:
    """Test MARL error handler."""

    @pytest.fixture
    def mock_recovery_manager(self):
        """Create mock recovery manager."""
        manager = Mock(spec=RecoveryStrategyManager)

        # Mock successful recovery strategy
        mock_strategy = Mock()
        mock_strategy.strategy_name = "test_strategy"
        mock_strategy.execute_recovery = AsyncMock(
            return_value=RecoveryResult(True, "test_strategy", 1.0, {"action": "test"})
        )

        manager.get_recovery_strategy = AsyncMock(return_value=mock_strategy)
        return manager

    @pytest.fixture
    def mock_error_analyzer(self):
        """Create mock error analyzer."""
        analyzer = Mock(spec=ErrorAnalyzer)
        analyzer.analyze_error = AsyncMock(
            return_value={
                "matching_patterns": [],
                "recommendations": ["test recommendation"],
            }
        )
        return analyzer

    def test_error_handler_initialization(self, mock_recovery_manager, mock_error_analyzer):
        """Test error handler initialization."""
        handler = MARLErrorHandler(
            recovery_manager=mock_recovery_manager,
            error_analyzer=mock_error_analyzer,
            max_recovery_attempts=2,
            recovery_timeout=10.0,
        )

        assert handler.recovery_manager == mock_recovery_manager
        assert handler.error_analyzer == mock_error_analyzer
        assert handler.max_recovery_attempts == 2
        assert handler.recovery_timeout == 10.0

    @pytest.mark.asyncio
    async def test_handle_marl_error(self, mock_recovery_manager, mock_error_analyzer):
        """Test handling MARL error."""
        handler = MARLErrorHandler(
            recovery_manager=mock_recovery_manager, error_analyzer=mock_error_analyzer
        )

        error = AgentError("Test error", "test_agent")
        result = await handler.handle_error(error)

        assert result["success"] is True
        assert result["error_id"] == error.error_id
        assert result["recovery_strategy"] == "test_strategy"

        # Verify error was removed from active errors after successful recovery
        assert error.error_id not in handler.active_errors

        # Verify analyzer was called
        mock_error_analyzer.analyze_error.assert_called_once_with(error)

    @pytest.mark.asyncio
    async def test_handle_generic_exception(self, mock_recovery_manager, mock_error_analyzer):
        """Test handling generic exception."""
        handler = MARLErrorHandler(
            recovery_manager=mock_recovery_manager, error_analyzer=mock_error_analyzer
        )

        # Test with generic exception
        exception = ValueError("Invalid value")
        context = {"agent_id": "test_agent"}

        result = await handler.handle_error(exception, context)

        assert result["success"] is True
        assert "error_id" in result

        # Verify error was converted to MARL error and then removed after successful recovery
        assert len(handler.active_errors) == 0  # Should be removed after successful recovery

    @pytest.mark.asyncio
    async def test_recovery_failure(self, mock_error_analyzer):
        """Test handling recovery failure."""
        # Mock failing recovery manager
        mock_recovery_manager = Mock(spec=RecoveryStrategyManager)
        mock_strategy = Mock()
        mock_strategy.strategy_name = "failing_strategy"
        mock_strategy.execute_recovery = AsyncMock(
            return_value=RecoveryResult(False, "failing_strategy", 1.0, error="Recovery failed")
        )
        mock_recovery_manager.get_recovery_strategy = AsyncMock(return_value=mock_strategy)

        handler = MARLErrorHandler(
            recovery_manager=mock_recovery_manager,
            error_analyzer=mock_error_analyzer,
            max_recovery_attempts=2,
        )

        error = AgentError("Test error", "test_agent")
        result = await handler.handle_error(error)

        assert result["success"] is False
        assert result["recovery_attempts"] == 2
        assert "last_error" in result

        # Error should still be active
        assert error.error_id in handler.active_errors

    @pytest.mark.asyncio
    async def test_recovery_timeout(self, mock_error_analyzer):
        """Test recovery timeout handling."""
        # Mock slow recovery manager
        mock_recovery_manager = Mock(spec=RecoveryStrategyManager)
        mock_strategy = Mock()
        mock_strategy.strategy_name = "slow_strategy"

        async def slow_recovery(error):
            await asyncio.sleep(2.0)  # Longer than timeout
            return RecoveryResult(True, "slow_strategy")

        mock_strategy.execute_recovery = slow_recovery
        mock_recovery_manager.get_recovery_strategy = AsyncMock(return_value=mock_strategy)

        handler = MARLErrorHandler(
            recovery_manager=mock_recovery_manager,
            error_analyzer=mock_error_analyzer,
            recovery_timeout=0.5,  # Short timeout
        )

        error = AgentError("Test error", "test_agent")
        result = await handler.handle_error(error)

        assert result["success"] is False
        assert "timeout" in result.get("last_error", "").lower()

    def test_error_classification(self):
        """Test error type classification."""
        handler = MARLErrorHandler()

        # Test agent error classification
        error_type = handler._classify_error_type(
            ValueError("Test error"), {"agent_id": "test_agent"}, None
        )
        assert error_type == "agent"

        # Test coordination error classification
        error_type = handler._classify_error_type(RuntimeError("coordination failed"), {}, None)
        assert error_type == "coordination"

        # Test timeout error classification
        error_type = handler._classify_error_type(TimeoutError("Operation timed out"), {}, None)
        assert error_type == "performance"

    @pytest.mark.asyncio
    async def test_recovery_callbacks(self, mock_recovery_manager, mock_error_analyzer):
        """Test recovery callbacks."""
        handler = MARLErrorHandler(
            recovery_manager=mock_recovery_manager, error_analyzer=mock_error_analyzer
        )

        callback_called = False
        callback_error = None
        callback_result = None

        def test_callback(error, result):
            nonlocal callback_called, callback_error, callback_result
            callback_called = True
            callback_error = error
            callback_result = result

        # Add callback
        handler.add_recovery_callback(test_callback)

        # Handle error (this should trigger callback)
        error = AgentError("Test error", "test_agent")

        # We need to manually trigger the callback since we're not running the full async flow
        await handler._notify_recovery_callbacks(error, {"success": True})

        # Verify callback was called
        assert callback_called is True
        assert callback_error == error
        assert callback_result == {"success": True}

        # Remove callback
        handler.remove_recovery_callback(test_callback)
        assert test_callback not in handler.recovery_callbacks

    def test_error_statistics(self, mock_recovery_manager, mock_error_analyzer):
        """Test error statistics tracking."""
        handler = MARLErrorHandler(
            recovery_manager=mock_recovery_manager, error_analyzer=mock_error_analyzer
        )

        # Initially empty
        stats = handler.get_error_statistics()
        assert stats["total_errors"] == 0

        # Record some errors manually
        error1 = AgentError("Error 1", "agent1")
        error2 = CoordinationError("Error 2")

        handler.error_statistics.record_error(error1)
        handler.error_statistics.record_error(error2)

        stats = handler.get_error_statistics()
        assert stats["total_errors"] == 2
        assert stats["errors_by_type"]["AgentError"] == 1
        assert stats["errors_by_type"]["CoordinationError"] == 1

    def test_active_errors_tracking(self, mock_recovery_manager, mock_error_analyzer):
        """Test active errors tracking."""
        handler = MARLErrorHandler(
            recovery_manager=mock_recovery_manager, error_analyzer=mock_error_analyzer
        )

        # Add some active errors
        error1 = AgentError("Error 1", "agent1")
        error2 = CoordinationError("Error 2")

        handler.active_errors[error1.error_id] = error1
        handler.active_errors[error2.error_id] = error2

        active_errors = handler.get_active_errors()
        assert len(active_errors) == 2

        # Check error dictionaries
        error_ids = [e["error_id"] for e in active_errors]
        assert error1.error_id in error_ids
        assert error2.error_id in error_ids

    def test_recovery_history(self, mock_recovery_manager, mock_error_analyzer):
        """Test recovery history tracking."""
        handler = MARLErrorHandler(
            recovery_manager=mock_recovery_manager, error_analyzer=mock_error_analyzer
        )

        # Add some history entries manually
        error = AgentError("Test error", "test_agent")
        recovery_result = {"success": True, "recovery_strategy": "test"}

        handler._update_recovery_history(error, recovery_result, 1.5)

        history = handler.get_recovery_history()
        assert len(history) == 1

        entry = history[0]
        assert entry["error_id"] == error.error_id
        assert entry["recovery_success"] is True
        assert entry["recovery_time"] == 1.5

        # Test history with limit
        limited_history = handler.get_recovery_history(limit=0)
        assert len(limited_history) == 0

    def test_clear_error_history(self, mock_recovery_manager, mock_error_analyzer):
        """Test clearing error history."""
        handler = MARLErrorHandler(
            recovery_manager=mock_recovery_manager, error_analyzer=mock_error_analyzer
        )

        # Add some data
        error = AgentError("Test error", "test_agent")
        handler.error_statistics.record_error(error)
        handler.recovery_history.append({"test": "data"})

        # Clear history
        handler.clear_error_history()

        assert handler.error_statistics.total_errors == 0
        assert len(handler.recovery_history) == 0

    @pytest.mark.asyncio
    async def test_shutdown(self, mock_recovery_manager, mock_error_analyzer):
        """Test error handler shutdown."""
        # Add shutdown methods to mocks
        mock_recovery_manager.shutdown = AsyncMock()
        mock_error_analyzer.shutdown = AsyncMock()

        handler = MARLErrorHandler(
            recovery_manager=mock_recovery_manager, error_analyzer=mock_error_analyzer
        )

        # Add some active errors
        error = AgentError("Test error", "test_agent")
        handler.active_errors[error.error_id] = error

        # Add callback
        handler.add_recovery_callback(lambda e, r: None)

        await handler.shutdown()

        # Verify cleanup
        assert len(handler.active_errors) == 0
        assert len(handler.recovery_callbacks) == 0

        # Verify component shutdown
        mock_recovery_manager.shutdown.assert_called_once()
        mock_error_analyzer.shutdown.assert_called_once()


class TestErrorHandlerFactory:
    """Test error handler factory."""

    def test_create_default(self):
        """Test creating default error handler."""
        handler = ErrorHandlerFactory.create_default()

        assert isinstance(handler, MARLErrorHandler)
        assert handler.max_recovery_attempts == 3
        assert handler.recovery_timeout == 30.0
        assert handler.enable_pattern_learning is True

    def test_create_with_config(self):
        """Test creating error handler with custom config."""
        config = {
            "max_recovery_attempts": 5,
            "recovery_timeout": 60.0,
            "enable_pattern_learning": False,
        }

        handler = ErrorHandlerFactory.create_with_config(config)

        assert handler.max_recovery_attempts == 5
        assert handler.recovery_timeout == 60.0
        assert handler.enable_pattern_learning is False

    def test_create_for_testing(self):
        """Test creating error handler for testing."""
        handler = ErrorHandlerFactory.create_for_testing()

        assert handler.max_recovery_attempts == 1
        assert handler.recovery_timeout == 5.0
        assert handler.enable_pattern_learning is False


if __name__ == "__main__":
    pytest.main([__file__])
