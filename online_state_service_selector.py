from typing import TYPE_CHECKING
from semantic_kernel.services.ai_service_selector import AIServiceSelector
import logging
from utils import internet

if TYPE_CHECKING:
    from semantic_kernel.functions.kernel_arguments import KernelArguments
    from semantic_kernel.functions.kernel_function import KernelFunction
    from semantic_kernel.kernel_types import AI_SERVICE_CLIENT_TYPE
    from semantic_kernel.services.ai_service_client_base import AIServiceClientBase
    from semantic_kernel.services.ai_service_selector import AIServiceSelector
    from semantic_kernel.services.kernel_services_extension import (
        KernelServicesExtension,
    )
    from semantic_kernel.connectors.ai.prompt_execution_settings import (
        PromptExecutionSettings,
    )

logger = logging.getLogger(__name__)


class OnlineStateServiceSelector(AIServiceSelector):
    def select_ai_service(
        self,
        kernel: "KernelServicesExtension",
        function: "KernelFunction",
        arguments: "KernelArguments",
        type_: type["AI_SERVICE_CLIENT_TYPE"]
        | tuple[type["AI_SERVICE_CLIENT_TYPE"], ...]
        | None = None,
    ) -> tuple["AIServiceClientBase", "PromptExecutionSettings"]:
        func_exec_settings = getattr(function, "prompt_execution_settings", None)
        logger.info(
            f"Function: {function.name}, prompt_execution_settings: {func_exec_settings}"
        )
        if internet() and "online" in func_exec_settings:
            return kernel.get_service("online"), func_exec_settings["online"]
        elif not internet() and "offline" in func_exec_settings:
            return kernel.get_service("offline"), func_exec_settings["offline"]
        return super().select_ai_service(kernel, function, arguments, type_)
