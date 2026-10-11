"""Robot CI Hardware Integration Readiness Audit.

Evaluates environment, transport, robot driver configuration, and safety interlocks
to determine physical hardware integration readiness without issuing motion commands
or making unsafe assumptions.

Safety Principles:
- Fails closed when required robot identity, driver, or safety limits are missing.
- Strictly read-only: NEVER issues motion commands during readiness audits.
- Never claims software stop requests (request_safe_stop) are certified hardware E-stops (STO).
- Accurately distinguishes actual hardware sensors from derived kinematic estimates.
- Preserves mock and PyBullet simulation workflows independently of ROS 2.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from robot_ci.simulation.safety import HardwareConfig
from robot_ci.transport.base import RobotTransport
from robot_ci.transport.fake_transport import FakeRobotTransport
from robot_ci.transport.ros2_adapter import is_ros2_available, ROS2RobotTransport


class HardwareReadinessStatus(str, Enum):
    """Overall hardware integration readiness status."""
    READY = "READY"
    READY_WITH_WARNINGS = "READY_WITH_WARNINGS"
    BLOCKED = "BLOCKED"


def check_python_packages() -> Dict[str, Dict[str, Any]]:
    """Check availability and versions of core and optional packages."""
    package_names = [
        ("numpy", True, "Kinematics, numerical matrices, and action vectors"),
        ("torch", False, "Neural network policy execution and inference"),
        ("cv2", False, "Computer vision trajectory extraction from video"),
        ("mcap", False, "Offline MCAP telemetry recording and playback"),
        ("rclpy", False, "ROS 2 Python client library (physical hardware runtime)"),
        ("sensor_msgs", False, "ROS 2 sensor messages (JointState)"),
        ("geometry_msgs", False, "ROS 2 geometry messages (PoseStamped)"),
    ]

    results: Dict[str, Dict[str, Any]] = {}
    for pkg, required, purpose in package_names:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed")
            results[pkg] = {
                "available": True,
                "version": str(version),
                "required": required,
                "purpose": purpose,
            }
        except (ImportError, ModuleNotFoundError):
            results[pkg] = {
                "available": False,
                "version": None,
                "required": required,
                "purpose": purpose,
            }
    return results


def check_ros2_environment() -> Dict[str, Any]:
    """Inspect system ROS 2 environment variables and rclpy availability."""
    ros_available = is_ros2_available()
    ros_distro = os.environ.get("ROS_DISTRO", None)
    ros_version = os.environ.get("ROS_VERSION", None)
    os_name = platform.system()
    os_release = platform.release()

    notes: List[str] = []
    if not ros_available:
        if os_name == "Windows":
            notes.append(
                "ROS 2 client library ('rclpy') is not available on this Windows host. "
                "Physical ROS 2 hardware drivers typically require an Ubuntu Linux host "
                "(e.g., Ubuntu 22.04 LTS for ROS 2 Humble) or a pre-configured ROS 2 container."
            )
        else:
            notes.append("ROS 2 Python bindings ('rclpy') are not installed in the active environment.")
    else:
        notes.append(f"ROS 2 Python bindings are available (ROS_DISTRO={ros_distro or 'unspecified'}).")

    return {
        "available": ros_available,
        "rclpy_installed": ros_available,
        "ros_distro": ros_distro,
        "ros_version": ros_version,
        "os_platform": f"{os_name} {os_release} ({platform.machine()})",
        "python_version": sys.version.split()[0],
        "notes": notes,
    }


@dataclass
class HardwareReadinessReport:
    """Structured report assessing physical hardware integration readiness."""
    report_id: str = field(default_factory=lambda: f"hw_ready_{uuid.uuid4().hex[:10]}")
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: HardwareReadinessStatus = HardwareReadinessStatus.BLOCKED
    
    # Environment & Packages
    python_packages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ros2_environment: Dict[str, Any] = field(default_factory=dict)
    
    # Transport Details
    configured_transport: Dict[str, Any] = field(default_factory=dict)
    is_fake_transport: bool = False
    
    # Robot & Driver Configuration
    robot_model: str = "unconfigured_arm"
    is_robot_model_configured: bool = False
    driver_configured: bool = False
    driver_type: Optional[str] = None
    
    # Safety Limits & Interlocks
    safety_configuration_verified: bool = False
    joint_limits_verified: bool = False
    workspace_bounds_verified: bool = False
    max_step_size_verified: bool = False
    enforce_workspace: bool = True
    enforce_joint_limits: bool = True
    
    # Stop Mechanisms
    stop_mechanisms: Dict[str, Any] = field(default_factory=dict)
    
    # Findings
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    prerequisites_for_live_execution: List[str] = field(default_factory=list)
    disclaimers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    def save(self, filepath: Optional[str | Path] = None) -> Path:
        if filepath is None:
            out_dir = Path("results/hardware_readiness")
            out_dir.mkdir(parents=True, exist_ok=True)
            filepath = out_dir / f"{self.report_id}.json"
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return filepath


def audit_hardware_readiness(
    config: Optional[HardwareConfig] = None,
    transport: Optional[RobotTransport] = None,
    allow_fake_transport_for_testing: bool = True,
) -> HardwareReadinessReport:
    """Perform a comprehensive, non-intrusive audit of hardware integration readiness.
    
    IMPORTANT: This function is strictly read-only. It NEVER sends motion commands,
    does NOT move any actuator, and fails closed whenever critical safety configuration
    or driver bindings are missing.

    Args:
        config: HardwareConfig defining robot model, joint bounds, workspace limits, etc.
        transport: RobotTransport instance (e.g. FakeRobotTransport or ROS2RobotTransport).
        allow_fake_transport_for_testing: If True, FakeRobotTransport results in
            READY_WITH_WARNINGS instead of BLOCKED when all other safety parameters pass.

    Returns:
        HardwareReadinessReport detailing status, blockers, warnings, and prerequisites.
    """
    report = HardwareReadinessReport()
    report.python_packages = check_python_packages()
    report.ros2_environment = check_ros2_environment()

    blockers: List[str] = []
    warnings: List[str] = []

    # 1. Package verification
    for pkg_name, pkg_info in report.python_packages.items():
        if pkg_info["required"] and not pkg_info["available"]:
            blockers.append(f"Required package '{pkg_name}' is not installed ({pkg_info['purpose']}).")
        elif not pkg_info["required"] and not pkg_info["available"] and pkg_name in ("cv2", "mcap"):
            warnings.append(f"Optional package '{pkg_name}' is not installed ({pkg_info['purpose']}).")

    # 2. Hardware Configuration Verification
    active_config = config or HardwareConfig()
    report.robot_model = active_config.robot_name
    report.driver_type = active_config.driver_type
    report.driver_configured = getattr(active_config, "driver_configured", False)
    report.enforce_workspace = active_config.enforce_workspace
    report.enforce_joint_limits = active_config.enforce_joint_limits

    # Robot Model identity verification
    is_model_valid = bool(
        active_config.robot_name
        and active_config.robot_name not in ("unconfigured_arm", "unknown_robot", "none", "placeholder")
    )
    report.is_robot_model_configured = is_model_valid
    if not is_model_valid:
        blockers.append(
            f"Robot model is unconfigured (found: '{active_config.robot_name}'). "
            "Real hardware driver requires confirmed robot manufacturer, model, kinematics, and URDF."
        )

    # Driver verification
    if not report.driver_configured:
        blockers.append(
            "Robot driver is unconfigured. "
            "A validated vendor driver, ros2_control hardware interface, or physical transport driver is required."
        )

    # 3. Critical Safety Configuration Verification
    # Joint limits
    has_joint_names = bool(active_config.joint_names and len(active_config.joint_names) > 0)
    has_joint_limits = bool(active_config.joint_limits and len(active_config.joint_limits) > 0)
    joint_limits_valid = False

    if has_joint_names and has_joint_limits:
        joint_limits_valid = True
        for jname in active_config.joint_names:
            if jname not in active_config.joint_limits:
                joint_limits_valid = False
                blockers.append(f"Joint '{jname}' lacks configured lower/upper limits.")
            else:
                lower, upper = active_config.joint_limits[jname]
                if lower >= upper:
                    joint_limits_valid = False
                    blockers.append(f"Joint '{jname}' has invalid limit range: [{lower}, {upper}] (lower >= upper).")
    else:
        blockers.append("Critical safety configuration missing: robot joint_names and joint_limits must be configured.")

    report.joint_limits_verified = joint_limits_valid

    # Workspace bounds
    workspace_valid = False
    wb = active_config.workspace_bounds
    if wb is not None and len(wb) == 2 and len(wb[0]) == 3 and len(wb[1]) == 3:
        mins, maxs = wb[0], wb[1]
        if all(mins[i] < maxs[i] for i in range(3)):
            workspace_valid = True
        else:
            blockers.append(f"Critical safety configuration invalid: workspace bounds min must be strictly less than max: {wb}")
    else:
        blockers.append("Critical safety configuration missing: workspace_bounds [[xmin, ymin, zmin], [xmax, ymax, zmax]] must be defined.")

    report.workspace_bounds_verified = workspace_valid

    # Max step size
    step_valid = False
    if active_config.max_step_size is not None and 0.0 < active_config.max_step_size <= 0.20:
        step_valid = True
    else:
        blockers.append(f"Critical safety configuration invalid: max_step_size ({active_config.max_step_size}m) must be > 0 and <= 0.20m.")

    report.max_step_size_verified = step_valid

    # Enforce flags
    if not active_config.enforce_workspace:
        blockers.append("Critical safety violation: enforce_workspace is False. Workspace boundary enforcement must be enabled.")
    if not active_config.enforce_joint_limits:
        blockers.append("Critical safety violation: enforce_joint_limits is False. Joint limit enforcement must be enabled.")

    report.safety_configuration_verified = (
        joint_limits_valid and workspace_valid and step_valid and active_config.enforce_workspace and active_config.enforce_joint_limits
    )

    # 4. Transport Inspection
    if transport is None:
        report.configured_transport = {
            "type": "NONE",
            "is_fake": False,
            "connected": False,
            "address": active_config.hardware_transport_address,
        }
        report.is_fake_transport = False
        blockers.append("No robot transport configured. Physical execution requires a connected transport instance.")
    else:
        transport_name = getattr(transport, "transport_name", transport.__class__.__name__)
        is_fake = isinstance(transport, FakeRobotTransport) or ("fake" in transport_name.lower())
        report.is_fake_transport = is_fake
        is_conn = transport.is_connected()

        report.configured_transport = {
            "type": transport_name,
            "is_fake": is_fake,
            "connected": is_conn,
            "address": active_config.hardware_transport_address,
        }

        if is_fake:
            warnings.append(
                "Configured transport is FakeRobotTransport. All motions and readings are simulated in software. "
                "No physical hardware commands will be dispatched."
            )
        else:
            # Real transport (e.g. ROS 2 or custom network socket)
            if isinstance(transport, ROS2RobotTransport):
                if not report.ros2_environment["available"]:
                    blockers.append(
                        "ROS 2 transport is configured, but ROS 2 runtime ('rclpy') is unavailable on this host."
                    )
            if not is_conn:
                blockers.append(f"Hardware transport '{transport_name}' is not currently connected.")

    # 5. Stop Mechanisms & Safety Distinction Audit
    transport_has_safe_stop = hasattr(transport, "request_safe_stop") if transport else False
    transport_estop_confirmed = False
    if transport and hasattr(transport, "is_safe_stop_confirmed"):
        try:
            transport_estop_confirmed = bool(transport.is_safe_stop_confirmed())
        except Exception:
            transport_estop_confirmed = False

    report.stop_mechanisms = {
        "software_stop_available": transport_has_safe_stop,
        "hardware_estop_confirmed": transport_estop_confirmed,
        "software_stop_description": (
            "Software deceleration request dispatched via transport message/service (request_safe_stop). "
            "Relies on OS scheduling, network stack, and robot controller software. Not fail-safe against power loss."
        ),
        "hardware_estop_description": (
            "Hardware-enforced Emergency Stop / Safe Torque Off (STO) circuit. "
            "Direct electromechanical or Category 3/4 PL d safety relay disconnecting actuator power independently of software."
        ),
    }

    if not transport_has_safe_stop:
        blockers.append("Transport does not support the request_safe_stop() software stop interface.")
    else:
        warnings.append(
            "Software stop (request_safe_stop) is available, but hardware-enforced Safe Torque Off (STO) "
            "must be physically wired and independently verified by an on-site safety operator."
        )

    # 6. Prerequisites for live hardware execution
    report.prerequisites_for_live_execution = [
        "1. Confirm physical robot manufacturer, exact model, kinematics/DH parameters, and degrees of freedom.",
        "2. Confirm robot communication interface (Ethernet/IP, EtherCAT, Modbus, or USB) and hardware controller firmware version.",
        "3. Provide confirmed ROS 2 distribution (e.g., ROS 2 Humble/Iron on Ubuntu 22.04 LTS real-time kernel) or native driver.",
        "4. Configure and validate the vendor-supported driver package (e.g., ros2_control hardware interface or vendor SDK).",
        "5. Wire and verify certified physical Emergency Stop button and Safe Torque Off (STO) safety relays.",
        "6. Define and calibrate physical workspace boundary bounding box (meters) and joint coordinate reference frames.",
        "7. Establish closed-loop physical completion verification tolerance epsilon (rad / m) before enabling closed-loop execution.",
    ]

    # 7. Safe Disclaimers
    report.disclaimers = [
        "Readiness checks are strictly non-intrusive and never issue motion commands.",
        "Package presence or successful module import does NOT prove a physical robot is wired, powered, or safe.",
        "Software stop requests (request_safe_stop) are software messages and DO NOT substitute for a hardware E-Stop (STO).",
        "All telemetry fields are categorized as ACTUAL_SENSOR (direct encoders) or DERIVED_ESTIMATE (kinematics/vision).",
        "Physical execution is DISABLED BY DEFAULT in Robot CI.",
    ]

    # 8. Determine Overall Status
    report.blockers = blockers
    report.warnings = warnings

    if len(blockers) > 0:
        report.status = HardwareReadinessStatus.BLOCKED
    elif report.is_fake_transport or len(warnings) > 0:
        report.status = HardwareReadinessStatus.READY_WITH_WARNINGS
    else:
        report.status = HardwareReadinessStatus.READY

    return report


def format_readiness_report_text(report: HardwareReadinessReport) -> str:
    """Format the hardware readiness report as a clear, human-readable terminal output."""
    lines: List[str] = []
    bar = "=" * 76
    subbar = "-" * 76

    status_str = report.status.value
    lines.append(bar)
    lines.append(f"  ROBOT CI -- Physical Hardware Integration Readiness Audit")
    lines.append(bar)
    lines.append(f"  Report ID:       {report.report_id}")
    lines.append(f"  Timestamp:       {report.timestamp}")
    lines.append(f"  Readiness Status: [{status_str}]")
    lines.append(subbar)

    # Section 1: Python & Package Availability
    lines.append("  1. ENVIRONMENT & PACKAGE AVAILABILITY:")
    for pkg, info in report.python_packages.items():
        req_tag = "(Required)" if info["required"] else "(Optional)"
        if info["available"]:
            lines.append(f"     [OK]   {pkg:<15} {info['version']:<12} {req_tag:<10} - {info['purpose']}")
        else:
            lines.append(f"     [FAIL] {pkg:<15} {'NOT INSTALLED':<12} {req_tag:<10} - {info['purpose']}")

    # Section 2: ROS 2 Runtime
    lines.append("\n  2. ROS 2 RUNTIME & PLATFORM:")
    ros_info = report.ros2_environment
    lines.append(f"     OS Platform:        {ros_info.get('os_platform', 'unknown')}")
    lines.append(f"     Python Version:     {ros_info.get('python_version', 'unknown')}")
    lines.append(f"     ROS 2 Available:    {ros_info.get('available')}")
    lines.append(f"     ROS_DISTRO:         {ros_info.get('ros_distro') or 'Not set'}")
    for n in ros_info.get("notes", []):
        lines.append(f"     Note: {n}")

    # Section 3: Transport & Driver Configuration
    lines.append("\n  3. TRANSPORT & DRIVER CONFIGURATION:")
    tc = report.configured_transport
    lines.append(f"     Configured Transport: {tc.get('type', 'NONE')}")
    lines.append(f"     Transport Mode:       {'SIMULATED / FAKE' if report.is_fake_transport else 'REAL HARDWARE / EXTERNAL'}")
    lines.append(f"     Transport Connected:  {tc.get('connected', False)}")
    lines.append(f"     Transport Address:    {tc.get('address') or 'Not configured'}")
    lines.append(f"     Robot Model:          {report.robot_model} (Configured: {report.is_robot_model_configured})")
    lines.append(f"     Driver Configured:    {report.driver_configured} (Type: {report.driver_type or 'None'})")

    # Section 4: Safety Configuration & Stop Mechanisms
    lines.append("\n  4. SAFETY CONFIGURATION & LIMITS:")
    lines.append(f"     Safety Config Status: {'VERIFIED' if report.safety_configuration_verified else 'INCOMPLETE / INVALID'}")
    lines.append(f"     Joint Limits:         {'VERIFIED' if report.joint_limits_verified else 'NOT VERIFIED'}")
    lines.append(f"     Workspace Bounds:     {'VERIFIED' if report.workspace_bounds_verified else 'NOT VERIFIED'}")
    lines.append(f"     Max Step Size:        {'VERIFIED' if report.max_step_size_verified else 'NOT VERIFIED'}")
    lines.append(f"     Enforce Workspace:    {report.enforce_workspace}")
    lines.append(f"     Enforce Joint Limits: {report.enforce_joint_limits}")

    # Stop mechanisms
    sm = report.stop_mechanisms
    lines.append("\n  5. STOP MECHANISMS (SOFTWARE REQUEST vs HARDWARE E-STOP):")
    lines.append(f"     Software Stop API:    {'AVAILABLE (request_safe_stop)' if sm.get('software_stop_available') else 'UNAVAILABLE'}")
    lines.append(f"     Hardware E-Stop (STO):{'CONFIRMED (is_safe_stop_confirmed)' if sm.get('hardware_estop_confirmed') else 'UNCONFIRMED / NOT DETECTED'}")
    lines.append(f"     Safety Note:          Software requests brake via controller logic; physical E-Stops require STO circuitry.")

    # Section 5: Blockers
    lines.append("\n" + subbar)
    if report.blockers:
        lines.append(f"  BLOCKERS PREVENTING PHYSICAL HARDWARE EXECUTION ({len(report.blockers)}):")
        for i, b in enumerate(report.blockers, 1):
            lines.append(f"    [{i}] [BLOCKER] {b}")
    else:
        lines.append("  BLOCKERS: None (All prerequisites satisfied)")

    # Section 6: Warnings
    if report.warnings:
        lines.append(f"\n  WARNINGS ({len(report.warnings)}):")
        for i, w in enumerate(report.warnings, 1):
            lines.append(f"    [{i}] [WARNING] {w}")

    # Section 7: Prerequisites for live execution
    lines.append(f"\n  PREREQUISITES BEFORE PHYSICAL ROBOT MOTION:")
    for prereq in report.prerequisites_for_live_execution:
        lines.append(f"    {prereq}")

    lines.append(bar)
    lines.append(f"  FINAL VERDICT: [{status_str}]")
    if report.status == HardwareReadinessStatus.BLOCKED:
        lines.append("  Physical execution is BLOCKED. Resolve all blockers above before attempting connection.")
    elif report.status == HardwareReadinessStatus.READY_WITH_WARNINGS:
        lines.append("  Safe for software/fake transport testing. Physical robot execution requires resolving warnings.")
    else:
        lines.append("  Hardware environment and safety checks fully satisfied.")
    lines.append(bar)

    return "\n".join(lines)
