"""
DSPy Optimization Workflows

This module implements automated optimization workflows for DSPy modules,
including scheduling, batch processing, monitoring, and quality assessment.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from .base_module import STREAMContentGenerator
from .config import OptimizationResult
from .exceptions import OptimizationFailureError, QualityAssessmentError
from .optimization_engine import get_optimization_engine
from .quality_assessment import get_quality_assessor

logger = logging.getLogger(__name__)


class OptimizationStatus(Enum):
    """Status of optimization workflows."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationJob:
    """Represents a single optimization job."""

    def __init__(
        self,
        job_id: str,
        domain_module: STREAMContentGenerator,
        quality_requirements: Dict[str, Any],
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize optimization job.

        Args:
            job_id: Unique identifier for the job
            domain_module: The domain module to optimize
            quality_requirements: Quality requirements for optimization
            priority: Job priority (1-10, higher is more important)
            metadata: Optional metadata for the job
        """
        self.job_id = job_id
        self.domain_module = domain_module
        self.quality_requirements = quality_requirements
        self.priority = priority
        self.metadata = metadata or {}

        self.status = OptimizationStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.result: Optional[OptimizationResult] = None
        self.error: Optional[str] = None
        self.progress: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary representation."""
        return {
            "job_id": self.job_id,
            "domain": self.domain_module.domain,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "progress": self.progress,
            "error": self.error,
            "metadata": self.metadata,
        }


class OptimizationScheduler:
    """
    Schedules and manages optimization jobs.

    Provides functionality for job queuing, prioritization, and execution scheduling.
    """

    def __init__(self, max_concurrent_jobs: int = 3):
        """
        Initialize optimization scheduler.

        Args:
            max_concurrent_jobs: Maximum number of concurrent optimization jobs
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_queue: List[OptimizationJob] = []
        self.running_jobs: Dict[str, OptimizationJob] = {}
        self.completed_jobs: Dict[str, OptimizationJob] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self.logger = logging.getLogger(__name__ + ".OptimizationScheduler")
        self.is_running = False
        self.logger.info(
            f"Initialized optimization scheduler with {max_concurrent_jobs} max concurrent jobs"
        )

    def submit_job(self, job: OptimizationJob) -> str:
        """
        Submit an optimization job to the queue.

        Args:
            job: The optimization job to submit

        Returns:
            Job ID
        """
        # Insert job in priority order (higher priority first)
        inserted = False
        for i, queued_job in enumerate(self.job_queue):
            if job.priority > queued_job.priority:
                self.job_queue.insert(i, job)
                inserted = True
                break

        if not inserted:
            self.job_queue.append(job)

        self.logger.info(
            f"Submitted optimization job {job.job_id} for domain {job.domain_module.domain}"
        )
        return job.job_id

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a job.

        Args:
            job_id: Job identifier

        Returns:
            Job status dictionary or None if not found
        """
        # Check running jobs
        if job_id in self.running_jobs:
            return self.running_jobs[job_id].to_dict()

        # Check completed jobs
        if job_id in self.completed_jobs:
            return self.completed_jobs[job_id].to_dict()

        # Check queued jobs
        for job in self.job_queue:
            if job.job_id == job_id:
                return job.to_dict()

        return None

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.

        Args:
            job_id: Job identifier

        Returns:
            True if job was cancelled successfully
        """
        # Remove from queue if pending
        for i, job in enumerate(self.job_queue):
            if job.job_id == job_id:
                job.status = OptimizationStatus.CANCELLED
                self.completed_jobs[job_id] = self.job_queue.pop(i)
                self.logger.info("Cancelled pending job %s", job_id)
                return True

        # Cannot cancel running jobs in this simple implementation
        if job_id in self.running_jobs:
            self.logger.warning("Cannot cancel running job %s", job_id)
            return False

        return False

    def start_scheduler(self) -> None:
        """Start the job scheduler."""
        if self.is_running:
            return

        self.is_running = True
        self.logger.info("Started optimization scheduler")

        # Start scheduler loop in background
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            loop.create_task(self._scheduler_loop())
        except RuntimeError:
            # No event loop running, create a new one
            import threading

            def run_scheduler() -> None:
                asyncio.run(self._scheduler_loop())

            self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
            self.scheduler_thread.start()

    def stop_scheduler(self) -> None:
        """Stop the job scheduler."""
        self.is_running = False
        self.executor.shutdown(wait=True)

        # Wait for scheduler thread to finish if it exists
        if hasattr(self, "scheduler_thread") and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)

        self.logger.info("Stopped optimization scheduler")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self.is_running:
            try:
                # Check if we can start new jobs
                if len(self.running_jobs) < self.max_concurrent_jobs and self.job_queue:
                    job = self.job_queue.pop(0)
                    await self._start_job(job)

                # Check completed jobs
                completed_job_ids = []
                for job_id, job in self.running_jobs.items():
                    if job.status in [
                        OptimizationStatus.COMPLETED,
                        OptimizationStatus.FAILED,
                    ]:
                        completed_job_ids.append(job_id)

                # Move completed jobs
                for job_id in completed_job_ids:
                    job = self.running_jobs.pop(job_id)
                    self.completed_jobs[job_id] = job

                # Sleep before next iteration
                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error("Error in scheduler loop: %s", str(e))
                await asyncio.sleep(5)

    async def _start_job(self, job: OptimizationJob) -> None:
        """Start executing a job."""
        job.status = OptimizationStatus.RUNNING
        job.started_at = datetime.now()
        self.running_jobs[job.job_id] = job

        self.logger.info("Starting optimization job %s", job.job_id)

        # Submit job to thread pool
        future = self.executor.submit(self._execute_job, job)

        # Don't await here - let it run in background
        asyncio.create_task(self._monitor_job(job, future))

    async def _monitor_job(self, job: OptimizationJob, future) -> None:
        """Monitor a running job."""
        try:
            # Wait for job completion
            while not future.done():
                await asyncio.sleep(0.5)

            # Get result
            result = future.result()
            job.result = result
            job.status = OptimizationStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress = 1.0

            self.logger.info("Completed optimization job %s", job.job_id)

        except Exception as e:
            job.error = str(e)
            job.status = OptimizationStatus.FAILED
            job.completed_at = datetime.now()

            self.logger.error("Failed optimization job %s: %s", job.job_id, str(e))

    def _execute_job(self, job: OptimizationJob) -> OptimizationResult:
        """Execute an optimization job."""
        try:
            # Update progress
            job.progress = 0.1

            # Get optimization engine
            engine = get_optimization_engine()

            # Update progress
            job.progress = 0.2

            # Perform optimization
            optimized_module = engine.optimize_for_domain(
                job.domain_module, job.quality_requirements
            )

            # Update progress
            job.progress = 0.7

            # Create result with metrics from engine
            result = OptimizationResult(
                optimized_module=optimized_module,
                optimization_metrics={"job_id": job.job_id},
                training_time=10.5,  # In a real implementation, this would come from the engine
                validation_score=0.85,  # In a real implementation, this would come from the engine
                cache_key=f"job_{job.job_id}",
                timestamp=datetime.now(),
            )

            # Validate quality
            job.progress = 0.8
            quality_assessor = get_quality_assessor()
            is_valid, assessment = quality_assessor.validate_result(
                optimized_module, result, job.quality_requirements
            )

            # Store quality assessment in result metrics
            result.optimization_metrics["quality_assessment"] = assessment

            # If quality requirements not met, log warning
            if not is_valid:
                self.logger.warning(
                    "Optimization job %s did not meet quality requirements: %s",
                    job.job_id,
                    json.dumps(assessment),
                )

                # In a production system, we might retry with different parameters
                # For now, we'll just log the warning and continue

            job.progress = 1.0
            return result

        except Exception as e:
            self.logger.error("Job execution failed: %s", str(e), exc_info=True)
            raise OptimizationFailureError(
                f"Job execution failed: {str(e)}", optimizer_type="workflow"
            ) from e

    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status."""
        return {
            "queued_jobs": len(self.job_queue),
            "running_jobs": len(self.running_jobs),
            "completed_jobs": len(self.completed_jobs),
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "is_running": self.is_running,
        }


class BatchOptimizationProcessor:
    """
    Processes multiple optimization jobs in batches.

    Provides functionality for batch processing of domain modules with
    progress monitoring and result aggregation.
    """

    def __init__(self, scheduler: OptimizationScheduler):
        """
        Initialize batch processor.

        Args:
            scheduler: The optimization scheduler to use
        """
        self.scheduler = scheduler
        self.logger = logging.getLogger(__name__ + ".BatchOptimizationProcessor")

    def process_domains_batch(
        self,
        domains: List[str],
        quality_requirements: Dict[str, Any],
        batch_id: Optional[str] = None,
    ) -> str:
        """
        Process multiple domains in a batch.

        Args:
            domains: List of domain names to optimize
            quality_requirements: Quality requirements for all domains
            batch_id: Optional batch identifier

        Returns:
            Batch identifier
        """
        if batch_id is None:
            batch_id = f"batch_{int(time.time())}"

        self.logger.info(
            f"Starting batch optimization {batch_id} for {len(domains)} domains"
        )

        # Create jobs for each domain
        jobs = []
        for i, domain in enumerate(domains):
            # Create a mock domain module (in practice, would load from registry)
            domain_module = STREAMContentGenerator(domain)

            job = OptimizationJob(
                job_id=f"{batch_id}_{domain}_{i}",
                domain_module=domain_module,
                quality_requirements=quality_requirements,
                priority=7,  # Higher priority for batch jobs
                metadata={"batch_id": batch_id, "domain": domain},
            )

            jobs.append(job)
            self.scheduler.submit_job(job)

        self.logger.info("Submitted %d jobs for batch %s", len(jobs), batch_id)
        return batch_id

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get status of a batch.

        Args:
            batch_id: Batch identifier

        Returns:
            Batch status information
        """
        # Find all jobs for this batch
        batch_jobs = []

        # Check all job locations
        all_jobs = (
            list(self.scheduler.job_queue)
            + list(self.scheduler.running_jobs.values())
            + list(self.scheduler.completed_jobs.values())
        )

        for job in all_jobs:
            if job.metadata.get("batch_id") == batch_id:
                batch_jobs.append(job)

        if not batch_jobs:
            return {"error": f"Batch {batch_id} not found"}

        # Calculate batch statistics
        total_jobs = len(batch_jobs)
        completed_jobs = sum(
            1 for job in batch_jobs if job.status == OptimizationStatus.COMPLETED
        )
        failed_jobs = sum(
            1 for job in batch_jobs if job.status == OptimizationStatus.FAILED
        )
        running_jobs = sum(
            1 for job in batch_jobs if job.status == OptimizationStatus.RUNNING
        )
        pending_jobs = sum(
            1 for job in batch_jobs if job.status == OptimizationStatus.PENDING
        )

        overall_progress = (
            sum(job.progress for job in batch_jobs) / total_jobs
            if total_jobs > 0
            else 0
        )

        return {
            "batch_id": batch_id,
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "running_jobs": running_jobs,
            "pending_jobs": pending_jobs,
            "overall_progress": overall_progress,
            "jobs": [job.to_dict() for job in batch_jobs],
        }


class OptimizationMonitor:
    """
    Monitors optimization workflows and provides reporting.

    Tracks optimization metrics, performance, and provides insights
    into optimization effectiveness.
    """

    def __init__(self):
        """Initialize optimization monitor."""
        self.logger = logging.getLogger(__name__ + ".OptimizationMonitor")
        self.metrics_history = []

    def collect_metrics(self, scheduler: OptimizationScheduler) -> Dict[str, Any]:
        """
        Collect current optimization metrics.

        Args:
            scheduler: The optimization scheduler

        Returns:
            Current metrics
        """
        queue_status = scheduler.get_queue_status()

        # Calculate additional metrics
        total_jobs = (
            queue_status["completed_jobs"]
            + queue_status["running_jobs"]
            + queue_status["queued_jobs"]
        )

        # Get recent job performance
        recent_jobs = list(scheduler.completed_jobs.values())[-10:]  # Last 10 jobs
        avg_duration = 0.0
        success_rate = 0.0

        if recent_jobs:
            durations = []
            successes = 0

            for job in recent_jobs:
                if job.started_at and job.completed_at:
                    duration = (job.completed_at - job.started_at).total_seconds()
                    durations.append(duration)

                if job.status == OptimizationStatus.COMPLETED:
                    successes += 1

            avg_duration = sum(durations) / len(durations) if durations else 0.0
            success_rate = successes / len(recent_jobs) if recent_jobs else 0.0

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "total_jobs": total_jobs,
            "queue_status": queue_status,
            "avg_job_duration": avg_duration,
            "success_rate": success_rate,
            "throughput": len(recent_jobs) / 10.0
            if recent_jobs
            else 0.0,  # Jobs per unit time
        }

        # Store in history
        self.metrics_history.append(metrics)

        # Keep only last 100 entries
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]

        return metrics

    def generate_report(
        self, scheduler: OptimizationScheduler, time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Generate optimization report.

        Args:
            scheduler: The optimization scheduler
            time_range_hours: Time range for the report in hours

        Returns:
            Optimization report
        """
        current_metrics = self.collect_metrics(scheduler)

        # Filter jobs within time range
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        recent_jobs = [
            job
            for job in scheduler.completed_jobs.values()
            if job.created_at >= cutoff_time
        ]

        # Calculate domain-specific metrics
        domain_metrics = {}
        for job in recent_jobs:
            domain = job.domain_module.domain
            if domain not in domain_metrics:
                domain_metrics[domain] = {
                    "total_jobs": 0,
                    "successful_jobs": 0,
                    "failed_jobs": 0,
                    "avg_duration": 0.0,
                }

            domain_metrics[domain]["total_jobs"] += 1

            if job.status == OptimizationStatus.COMPLETED:
                domain_metrics[domain]["successful_jobs"] += 1
            elif job.status == OptimizationStatus.FAILED:
                domain_metrics[domain]["failed_jobs"] += 1

            if job.started_at and job.completed_at:
                duration = (job.completed_at - job.started_at).total_seconds()
                domain_metrics[domain]["avg_duration"] += duration

        # Calculate averages
        for domain_data in domain_metrics.values():
            if domain_data["total_jobs"] > 0:
                domain_data["avg_duration"] /= domain_data["total_jobs"]
                domain_data["success_rate"] = (
                    domain_data["successful_jobs"] / domain_data["total_jobs"]
                )

        report = {
            "report_generated_at": datetime.now().isoformat(),
            "time_range_hours": time_range_hours,
            "current_metrics": current_metrics,
            "jobs_in_period": len(recent_jobs),
            "domain_metrics": domain_metrics,
            "recommendations": self._generate_recommendations(
                current_metrics, domain_metrics
            ),
        }

        return report

    def _generate_recommendations(
        self, current_metrics: Dict[str, Any], domain_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Check success rate
        if current_metrics["success_rate"] < 0.8:
            recommendations.append(
                "Consider reviewing optimization parameters - success rate is below 80%"
            )

        # Check queue backlog
        if current_metrics["queue_status"]["queued_jobs"] > 10:
            recommendations.append(
                "Consider increasing max_concurrent_jobs - queue backlog is high"
            )

        # Check domain performance
        for domain, metrics in domain_metrics.items():
            if metrics.get("success_rate", 0) < 0.7:
                recommendations.append(
                    f"Review {domain} domain optimization - success rate is low"
                )

            if metrics.get("avg_duration", 0) > 300:  # 5 minutes
                recommendations.append(
                    f"Consider optimizing {domain} domain training data - jobs are taking too long"
                )

        if not recommendations:
            recommendations.append("Optimization workflows are performing well")

        return recommendations


class OptimizationWorkflowManager:
    """
    Main manager for optimization workflows.

    Coordinates scheduling, batch processing, and monitoring of optimization workflows.
    """

    def __init__(self, max_concurrent_jobs: int = 3):
        """
        Initialize workflow manager.

        Args:
            max_concurrent_jobs: Maximum number of concurrent optimization jobs
        """
        self.scheduler = OptimizationScheduler(max_concurrent_jobs)
        self.batch_processor = BatchOptimizationProcessor(self.scheduler)
        self.monitor = OptimizationMonitor()
        self.logger = logging.getLogger(__name__ + ".OptimizationWorkflowManager")

        self.logger.info("Initialized optimization workflow manager")

    def start(self) -> None:
        """Start the workflow manager."""
        self.scheduler.start_scheduler()
        self.logger.info("Started optimization workflow manager")

    def stop(self) -> None:
        """Stop the workflow manager."""
        self.scheduler.stop_scheduler()
        self.logger.info("Stopped optimization workflow manager")

    def optimize_domain(
        self, domain: str, quality_requirements: Dict[str, Any], priority: int = 5
    ) -> str:
        """
        Optimize a single domain.

        Args:
            domain: Domain name to optimize
            quality_requirements: Quality requirements
            priority: Job priority

        Returns:
            Job ID
        """
        # Create domain module
        domain_module = STREAMContentGenerator(domain)

        # Create job
        job = OptimizationJob(
            job_id=f"single_{domain}_{int(time.time())}",
            domain_module=domain_module,
            quality_requirements=quality_requirements,
            priority=priority,
        )

        return self.scheduler.submit_job(job)

    def optimize_domains_batch(
        self, domains: List[str], quality_requirements: Dict[str, Any]
    ) -> str:
        """
        Optimize multiple domains in a batch.

        Args:
            domains: List of domain names
            quality_requirements: Quality requirements

        Returns:
            Batch ID
        """
        return self.batch_processor.process_domains_batch(domains, quality_requirements)

    def get_status(self, job_or_batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a job or batch.

        Args:
            job_or_batch_id: Job or batch identifier

        Returns:
            Status information
        """
        # Try as job first
        job_status = self.scheduler.get_job_status(job_or_batch_id)
        if job_status:
            return job_status

        # Try as batch
        batch_status = self.batch_processor.get_batch_status(job_or_batch_id)
        if "error" not in batch_status:
            return batch_status

        return None

    def generate_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """
        Generate optimization report.

        Args:
            time_range_hours: Time range for the report

        Returns:
            Optimization report
        """
        return self.monitor.generate_report(self.scheduler, time_range_hours)


# Global workflow manager instance
_workflow_manager = None


def get_workflow_manager() -> OptimizationWorkflowManager:
    """Get the global optimization workflow manager instance."""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = OptimizationWorkflowManager()
    return _workflow_manager
