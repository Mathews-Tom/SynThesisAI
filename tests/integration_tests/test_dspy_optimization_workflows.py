"""
Integration tests for DSPy optimization workflows.

Tests the complete optimization workflow including scheduling,
batch processing, monitoring, and quality assessment.
"""

import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from core.dspy.base_module import STREAMContentGenerator
from core.dspy.config import OptimizationResult
from core.dspy.optimization_workflows import (
    BatchOptimizationProcessor,
    OptimizationJob,
    OptimizationMonitor,
    OptimizationScheduler,
    OptimizationStatus,
    OptimizationWorkflowManager,
    get_workflow_manager,
)


class TestOptimizationWorkflowsIntegration:
    """Integration tests for optimization workflows."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.scheduler = OptimizationScheduler(max_concurrent_jobs=2)
        self.batch_processor = BatchOptimizationProcessor(self.scheduler)
        self.monitor = OptimizationMonitor()
        self.workflow_manager = OptimizationWorkflowManager(max_concurrent_jobs=2)

    def teardown_method(self) -> None:
        """Clean up after tests."""
        if hasattr(self.scheduler, "stop_scheduler"):
            self.scheduler.stop_scheduler()
        if hasattr(self.workflow_manager, "stop"):
            self.workflow_manager.stop()

    def test_end_to_end_single_optimization(self):
        """Test complete single domain optimization workflow."""
        # Create domain module
        domain_module = STREAMContentGenerator("mathematics")
        quality_requirements = {"min_accuracy": 0.8, "max_training_time": 300}

        # Create optimization job
        job = OptimizationJob(
            job_id="test_job_1",
            domain_module=domain_module,
            quality_requirements=quality_requirements,
            priority=5,
        )

        # Submit job
        job_id = self.scheduler.submit_job(job)
        assert job_id == "test_job_1"

        # Check initial status
        status = self.scheduler.get_job_status(job_id)
        assert status is not None
        assert status["status"] == OptimizationStatus.PENDING.value
        assert status["domain"] == "mathematics"

        # Mock the optimization engine to avoid actual optimization
        with (
            patch("core.dspy.optimization_workflows.get_optimization_engine") as mock_engine,
            patch("core.dspy.optimization_workflows.get_quality_assessor") as mock_assessor,
        ):
            # Mock optimization engine
            mock_opt_engine = Mock()
            mock_optimized_module = STREAMContentGenerator("mathematics")
            mock_opt_engine.optimize_for_domain.return_value = mock_optimized_module
            mock_engine.return_value = mock_opt_engine

            # Mock quality assessor
            mock_quality_assessor = Mock()
            mock_assessment = {
                "domain": "mathematics",
                "overall_score": 0.9,
                "requirements_met": True,
            }
            mock_quality_assessor.validate_result.return_value = (True, mock_assessment)
            mock_assessor.return_value = mock_quality_assessor

            # Execute job directly (simulating scheduler execution)
            result = self.scheduler._execute_job(job)

            # Verify result
            assert isinstance(result, OptimizationResult)
            assert result.optimized_module == mock_optimized_module
            assert "quality_assessment" in result.optimization_metrics
            assert result.optimization_metrics["quality_assessment"] == mock_assessment

    def test_batch_optimization_workflow(self):
        """Test batch optimization workflow."""
        domains = ["mathematics", "science", "technology"]
        quality_requirements = {"min_accuracy": 0.8}

        # Process batch
        batch_id = self.batch_processor.process_domains_batch(domains, quality_requirements)

        # Check batch was created
        assert batch_id.startswith("batch_")

        # Check batch status
        batch_status = self.batch_processor.get_batch_status(batch_id)
        assert batch_status["batch_id"] == batch_id
        assert batch_status["total_jobs"] == 3
        assert batch_status["pending_jobs"] == 3
        assert batch_status["overall_progress"] == 0.0

        # Check individual jobs were created
        assert len(self.scheduler.job_queue) == 3
        for job in self.scheduler.job_queue:
            assert job.metadata["batch_id"] == batch_id
            assert job.domain_module.domain in domains
            assert job.priority == 7  # Higher priority for batch jobs

    def test_optimization_monitoring(self):
        """Test optimization monitoring and reporting."""
        # Create some completed jobs for monitoring
        completed_jobs = {}
        for i, domain in enumerate(["mathematics", "science"]):
            job = OptimizationJob(
                job_id=f"completed_job_{i}",
                domain_module=STREAMContentGenerator(domain),
                quality_requirements={"min_accuracy": 0.8},
            )
            job.status = OptimizationStatus.COMPLETED
            job.created_at = datetime.now() - timedelta(hours=1)
            job.started_at = datetime.now() - timedelta(minutes=30)
            job.completed_at = datetime.now() - timedelta(minutes=25)
            completed_jobs[job.job_id] = job

        self.scheduler.completed_jobs = completed_jobs

        # Collect metrics
        metrics = self.monitor.collect_metrics(self.scheduler)

        # Verify metrics
        assert "timestamp" in metrics
        assert metrics["total_jobs"] == 2
        assert metrics["success_rate"] == 1.0  # All jobs completed successfully
        assert metrics["avg_job_duration"] > 0

        # Generate report
        report = self.monitor.generate_report(self.scheduler, time_range_hours=2)

        # Verify report
        assert "report_generated_at" in report
        assert report["jobs_in_period"] == 2
        assert "domain_metrics" in report
        assert "mathematics" in report["domain_metrics"]
        assert "science" in report["domain_metrics"]
        assert "recommendations" in report

    def test_workflow_manager_integration(self):
        """Test complete workflow manager integration."""
        # Start workflow manager
        self.workflow_manager.start()

        # Optimize single domain
        job_id = self.workflow_manager.optimize_domain(
            "mathematics", {"min_accuracy": 0.8}, priority=7
        )

        # Check job was submitted
        assert job_id.startswith("single_mathematics_")
        status = self.workflow_manager.get_status(job_id)
        assert status is not None
        assert status["domain"] == "mathematics"
        assert status["priority"] == 7

        # Optimize batch of domains
        batch_id = self.workflow_manager.optimize_domains_batch(
            ["science", "technology"], {"min_accuracy": 0.9}
        )

        # Check batch was created
        assert batch_id.startswith("batch_")
        batch_status = self.workflow_manager.get_status(batch_id)
        assert batch_status is not None
        assert batch_status["total_jobs"] == 2

        # Generate report
        report = self.workflow_manager.generate_report(time_range_hours=1)
        assert "current_metrics" in report
        assert "domain_metrics" in report

        # Stop workflow manager
        self.workflow_manager.stop()

    def test_job_cancellation(self):
        """Test job cancellation functionality."""
        # Create and submit job
        domain_module = STREAMContentGenerator("mathematics")
        job = OptimizationJob(
            job_id="cancellable_job",
            domain_module=domain_module,
            quality_requirements={"min_accuracy": 0.8},
        )

        job_id = self.scheduler.submit_job(job)

        # Verify job is queued
        status = self.scheduler.get_job_status(job_id)
        assert status["status"] == OptimizationStatus.PENDING.value

        # Cancel job
        cancelled = self.scheduler.cancel_job(job_id)
        assert cancelled is True

        # Verify job is cancelled
        status = self.scheduler.get_job_status(job_id)
        assert status["status"] == OptimizationStatus.CANCELLED.value

        # Try to cancel non-existent job
        cancelled = self.scheduler.cancel_job("non_existent_job")
        assert cancelled is False

    def test_queue_status_tracking(self):
        """Test queue status tracking functionality."""
        # Initial queue status
        status = self.scheduler.get_queue_status()
        assert status["queued_jobs"] == 0
        assert status["running_jobs"] == 0
        assert status["completed_jobs"] == 0
        assert status["is_running"] is False

        # Add some jobs
        for i in range(3):
            job = OptimizationJob(
                job_id=f"test_job_{i}",
                domain_module=STREAMContentGenerator("mathematics"),
                quality_requirements={"min_accuracy": 0.8},
            )
            self.scheduler.submit_job(job)

        # Check updated status
        status = self.scheduler.get_queue_status()
        assert status["queued_jobs"] == 3
        assert status["running_jobs"] == 0
        assert status["completed_jobs"] == 0

    def test_priority_based_scheduling(self):
        """Test priority-based job scheduling."""
        # Create jobs with different priorities
        jobs = []
        for i, priority in enumerate([3, 8, 5, 9, 1]):
            job = OptimizationJob(
                job_id=f"priority_job_{i}",
                domain_module=STREAMContentGenerator("mathematics"),
                quality_requirements={"min_accuracy": 0.8},
                priority=priority,
            )
            jobs.append(job)
            self.scheduler.submit_job(job)

        # Check jobs are ordered by priority (highest first)
        queue_priorities = [job.priority for job in self.scheduler.job_queue]
        expected_priorities = [9, 8, 5, 3, 1]  # Sorted in descending order
        assert queue_priorities == expected_priorities

    def test_error_handling_in_workflows(self):
        """Test error handling in optimization workflows."""
        # Create job that will fail
        domain_module = STREAMContentGenerator("mathematics")
        job = OptimizationJob(
            job_id="failing_job",
            domain_module=domain_module,
            quality_requirements={"min_accuracy": 0.8},
        )

        # Mock optimization engine to raise exception
        with patch("core.dspy.optimization_workflows.get_optimization_engine") as mock_engine:
            mock_opt_engine = Mock()
            mock_opt_engine.optimize_for_domain.side_effect = Exception("Optimization failed")
            mock_engine.return_value = mock_opt_engine

            # Execute job and expect failure
            with pytest.raises(Exception):
                self.scheduler._execute_job(job)

    def test_global_workflow_manager_singleton(self):
        """Test global workflow manager singleton pattern."""
        # Get workflow manager instances
        manager1 = get_workflow_manager()
        manager2 = get_workflow_manager()

        # Should be the same instance
        assert manager1 is manager2
        assert isinstance(manager1, OptimizationWorkflowManager)


class TestOptimizationWorkflowsPerformance:
    """Performance tests for optimization workflows."""

    def test_concurrent_job_execution(self):
        """Test concurrent job execution performance."""
        scheduler = OptimizationScheduler(max_concurrent_jobs=3)

        # Create multiple jobs
        jobs = []
        for i in range(5):
            job = OptimizationJob(
                job_id=f"perf_job_{i}",
                domain_module=STREAMContentGenerator("mathematics"),
                quality_requirements={"min_accuracy": 0.8},
            )
            jobs.append(job)
            scheduler.submit_job(job)

        # Check that scheduler respects max concurrent jobs
        assert len(scheduler.job_queue) == 5
        assert len(scheduler.running_jobs) == 0

        # Verify queue status
        status = scheduler.get_queue_status()
        assert status["max_concurrent_jobs"] == 3
        assert status["queued_jobs"] == 5

    def test_batch_processing_scalability(self):
        """Test batch processing scalability."""
        scheduler = OptimizationScheduler(max_concurrent_jobs=2)
        batch_processor = BatchOptimizationProcessor(scheduler)

        # Process large batch
        domains = [f"domain_{i}" for i in range(10)]
        quality_requirements = {"min_accuracy": 0.8}

        start_time = time.time()
        batch_id = batch_processor.process_domains_batch(domains, quality_requirements)
        processing_time = time.time() - start_time

        # Should process quickly (just job creation, not execution)
        assert processing_time < 1.0

        # Check all jobs were created
        batch_status = batch_processor.get_batch_status(batch_id)
        assert batch_status["total_jobs"] == 10
        assert batch_status["pending_jobs"] == 10

    def test_monitoring_performance(self):
        """Test monitoring performance with many jobs."""
        scheduler = OptimizationScheduler()
        monitor = OptimizationMonitor()

        # Create many completed jobs
        for i in range(100):
            job = OptimizationJob(
                job_id=f"monitor_job_{i}",
                domain_module=STREAMContentGenerator(f"domain_{i % 6}"),
                quality_requirements={"min_accuracy": 0.8},
            )
            job.status = OptimizationStatus.COMPLETED
            job.created_at = datetime.now() - timedelta(minutes=i)
            job.started_at = datetime.now() - timedelta(minutes=i)
            job.completed_at = datetime.now() - timedelta(minutes=i - 1)
            scheduler.completed_jobs[job.job_id] = job

        # Measure metrics collection performance
        start_time = time.time()
        metrics = monitor.collect_metrics(scheduler)
        collection_time = time.time() - start_time

        # Should collect metrics quickly
        assert collection_time < 0.5
        assert metrics["total_jobs"] == 100

        # Measure report generation performance
        start_time = time.time()
        report = monitor.generate_report(scheduler, time_range_hours=24)
        report_time = time.time() - start_time

        # Should generate report quickly
        assert report_time < 1.0
        assert len(report["domain_metrics"]) == 6  # 6 different domains
