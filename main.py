from dataclasses import field
from backend import setup
from utils import internet
import mesop as me
from semantic_kernel.contents import (
    ChatHistory,
    StreamingChatMessageContent,
    TextContent,
)
from dotenv import load_dotenv

load_dotenv()

import debugpy

debugpy.listen(5678)
DEFAULT_FONT_FAMILY = '"Lucida Console", monospace'
SK_GRADIENT = "linear-gradient(90deg, #3D0D59, #801EAE, #C86FEC, #4A94FC) text"

DEFAULT_STYLE = {"font_family": DEFAULT_FONT_FAMILY}
DEFAULT_STYLE_WITH_GRADIENT = {
    "font_family": DEFAULT_FONT_FAMILY,
    "background": SK_GRADIENT,
    "color": "transparent",
}


EXAMPLES = [
    "How awesome is Semantic Kernel?",
    "what is a Chat Completion Agent and how do I create one?",
    "Does semantic kernel support ollama and if so how do I do that?",
]

kernel = setup()


@me.stateclass
class State:
    input: str
    temp_input: str
    output: str
    in_progress: bool
    chat_history: dict = field(default_factory=dict)


@me.page(
    path="/chat",
    title="Chat with Eduard using Semantic Kernel",
)
def page():
    with me.box(
        style=me.Style(
            display="grid", grid_template_rows="auto 1fr auto", height="100%"
        )
    ):
        # header
        header_text()
        # body
        with me.box(
            style=me.Style(
                margin=me.Margin.all("auto"),
                padding=me.Padding.all(30),
                width="60%",
            )
        ):
            chat_history()
            output()
            chat_input()
            example_row()
        # footer
        footer()


def header_text():
    buffer_width = "30%"
    with me.box(
        style=me.Style(
            padding=me.Padding(
                top=32,
                bottom=30,
            ),
            position="sticky",
            top=0,
            align_self="center",
            grid_template_columns=f"{buffer_width} 1fr 4fr {buffer_width}",
            display="grid",
            align_items="center",
            justify_items="center",
            background="#F0F4F9",
        )
    ):
        with me.box():
            pass
        me.image(src="/static/SK.svg")
        me.text(
            "Chat with Eduard",
            style=me.Style(
                text_align="center",
                font_size=42,
                font_weight=700,
                padding=me.Padding(left=40),
                **DEFAULT_STYLE_WITH_GRADIENT,
            ),
        )
        with me.box():
            pass


def example_row():
    is_mobile = me.viewport_size().width < 640
    with me.box(
        style=me.Style(
            font_family=DEFAULT_FONT_FAMILY,
            display="flex",
            flex_direction="column" if is_mobile else "row",
            justify_content="space-around",
            gap=24,
            margin=me.Margin(top=36),
        )
    ):
        for example in EXAMPLES:
            example_box(example, is_mobile)


def example_box(example: str, is_mobile: bool):
    with me.box(
        style=me.Style(
            width="100%" if is_mobile else 200,
            height=140,
            background="#F0F4F9",
            padding=me.Padding.all(16),
            font_weight=500,
            line_height="1.5",
            border_radius=16,
            cursor="pointer",
        ),
        key=example,
        on_click=click_example_box,
    ):
        me.text(example)


def click_example_box(e: me.ClickEvent):
    state = me.state(State)
    state.input = e.key


def chat_input():
    state = me.state(State)
    with me.box(
        style=me.Style(
            padding=me.Padding.all(8),
            background="white",
            display="flex",
            width="90%",
            justify_self="center",
            border=me.Border.all(me.BorderSide(width=0, style="solid", color="black")),
            border_radius=12,
            box_shadow="0 10px 20px #0000000a, 0 2px 6px #0000000a, 0 0 1px #0000000a",
            **DEFAULT_STYLE,
        )
    ):
        with me.box(
            style=me.Style(
                flex_grow=1,
            )
        ):
            me.native_textarea(
                value=state.input,
                autosize=True,
                min_rows=4,
                placeholder="Enter your prompt",
                style=me.Style(
                    padding=me.Padding(top=16, left=16),
                    background="white",
                    outline="none",
                    width="100%",
                    overflow_y="auto",
                    border=me.Border.all(
                        me.BorderSide(style="none"),
                    ),
                    **DEFAULT_STYLE,
                ),
                on_blur=textarea_on_blur,
            )
        with me.content_button(type="icon", on_click=click_send):
            me.icon("send")


def textarea_on_blur(e: me.InputBlurEvent):
    state = me.state(State)
    state.input = e.value


async def click_send(e: me.ClickEvent):
    state = me.state(State)
    if not state.input:
        return
    state.in_progress = True
    state.temp_input = input = state.input
    state.input = ""
    yield

    async for chunk in call_api(input):
        state.output += chunk
        yield
    state.in_progress = False
    state.output = ""
    state.temp_input = ""
    yield


async def call_api(input):
    state = me.state(State)
    if state.chat_history:
        chat_history = ChatHistory.model_validate(state.chat_history)
    else:
        chat_history = ChatHistory()
    chunks: list[StreamingChatMessageContent] = []
    async for response in kernel.invoke_stream(
        function_name="chat",
        plugin_name="chat",
        chat_history=chat_history,
        user_input=input,
    ):
        chunks.append(response[0])
        if response[0].content:
            yield response[0].content
    chat_history.add_user_message(input)
    full_msg: StreamingChatMessageContent = sum(chunks[1:], chunks[0])
    new_items = []
    for item in full_msg.items:
        if isinstance(item, TextContent):
            new_items.append(item)
    full_msg.items = new_items
    chat_history.add_message(full_msg)
    state.chat_history = chat_history.model_dump()


@me.component
def user_message(message: str):
    me.markdown(
        message,
        style=me.Style(
            background="#F0F4F9",
            padding=me.Padding.all(16),
            border_radius=16,
            margin=me.Margin(left=130, right=16, top=16, bottom=16),
            right=0,
            **DEFAULT_STYLE,
        ),
    )


@me.component
def text_avatar(*, label: str, background: str, color: str):
    me.text(
        label,
        style=me.Style(
            background=background,
            border_radius="50%",
            color=color,
            font_size=20,
            height=40,
            line_height="1",
            margin=me.Margin(top=16),
            padding=me.Padding(top=10),
            text_align="center",
            width="40px",
        ),
    )


@me.component
def assistant_message(message: str):
    with me.box(style=me.Style(display="flex", gap=15, margin=me.Margin.all(20))):
        text_avatar(
            background="#E8E4F9",
            color="gray",
            label="E",
        )

        # Bot message response
        with me.box(style=me.Style(display="flex", flex_direction="column")):
            me.markdown(
                message,
                style=me.Style(
                    background="#E0E0FF",
                    padding=me.Padding.all(16),
                    border_radius=16,
                    margin=me.Margin(left=16, right=32, top=16, bottom=16),
                    **DEFAULT_STYLE,
                ),
            )


def chat_history():
    state = me.state(State)
    if state.chat_history:
        chat_history = ChatHistory.model_validate(state.chat_history)
        for message in chat_history.messages:
            if message.role == "user":
                user_message(message.content)
            else:
                assistant_message(message.content)


def output():
    state = me.state(State)
    if state.temp_input and (state.output or state.in_progress):
        user_message(state.temp_input)
    if state.output:
        assistant_message(state.output)
    if state.in_progress:
        with me.box(style=me.Style(margin=me.Margin(top=16), justify_self="center")):
            me.progress_spinner()


def footer():
    with me.box(
        style=me.Style(
            position="sticky",
            bottom=0,
            padding=me.Padding.symmetric(vertical=16, horizontal=16),
            width="100%",
            background="#F0F4F9",
            font_size=14,
        )
    ):
        me.html(
            "Powered by <a href='https://github.com/microsoft/semantic-kernel' target='_blank'>Semantic Kernel!</>",
            style=me.Style(
                text_align="center",
                **DEFAULT_STYLE_WITH_GRADIENT,
            ),
        )
        me.text(
            f"Internet status: {"online" if internet() else "offline"}",
            style=me.Style(
                font_size=12,
                text_align="center",
                padding=me.Padding(top=8),
                **DEFAULT_STYLE_WITH_GRADIENT,
            ),
        )


# def sidebar():
#     state = me.state(State)
#     with me.box(
#         style=me.Style(
#             display="flex",
#             flex_direction="column",
#             flex_grow=1,
#         )
#     ):
#         with me.box(style=me.Style(display="flex", gap=20)):
#             menu_icon(icon="menu", tooltip="Menu", on_click=on_click_menu_icon)
#             if state.sidebar_expanded:
#                 me.text(
#                     _APP_TITLE,
#                     style=me.Style(margin=me.Margin(bottom=0, top=14)),
#                     type="headline-6",
#                 )

#         if state.sidebar_expanded:
#             menu_item(icon="add", label="New chat", on_click=on_click_new_chat)
#         else:
#             menu_icon(icon="add", tooltip="New chat", on_click=on_click_new_chat)

#         if state.sidebar_expanded:
#             pass
#             # history_pane()


# @me.component
# def menu_item(
#     *, icon: str, label: str, key: str = "", on_click: Callable | None = None
# ):
#     with me.box(on_click=on_click):
#         with me.box(
#             style=me.Style(
#                 background=me.theme_var("surface-container-high"),
#                 border_radius=20,
#                 cursor="pointer",
#                 display="inline-flex",
#                 gap=10,
#                 line_height=1,
#                 margin=me.Margin.all(10),
#                 padding=me.Padding(top=10, left=10, right=20, bottom=10),
#             ),
#         ):
#             me.icon(icon)
#             me.text(label, style=me.Style(height=24, line_height="24px"))
