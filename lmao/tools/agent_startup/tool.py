from lmao.plugins import PLUGIN_API_VERSION

PLUGIN = {
    "name": "agent_startup",
    "description": "Internal tool for agent startup operations (hidden from agents)",
    "api_version": PLUGIN_API_VERSION,
    "is_destructive": False,
    "allow_in_read_only": True,
    "allow_in_normal": True,
    "allow_in_yolo": True,
    "always_confirm": False,
    "visible_to_agent": False,  # This tool is hidden from agents
    "usage": ['{"tool":"agent_startup","target":"","args":""}'],
    "hooks": {"agent_startup": ["execute_startup_tools"]},
}


def run(
    target,
    args,
    base,
    extra_roots,
    skill_roots,
    task_manager=None,
    debug_logger=None,
    meta=None,
):
    """Internal startup tool - not exposed to agents."""
    if debug_logger:
        debug_logger.log("agent_startup.tool", f"executing startup with args={args}")
    return '{"tool":"agent_startup","success":true,"data":{"message":"Agent startup completed"}}'


def execute_startup_tools(context):
    """Hook handler for agent startup."""
    debug_logger = context.runtime_state.get("debug_logger")
    if debug_logger:
        debug_logger.log("agent_startup.hook", "executing startup hooks")
    # This could trigger policy/skills_guide tools internally
    return {"success": True, "message": "Startup hooks executed"}
