import streamlit as st
from streamlit_chat import message

from predict import load_model, get_prediction


model, tokenizer = load_model()


st.title("Streamlit STS : 두 문장 유사도 파악하기")
if "input" not in st.session_state:
    st.session_state["input_1"] = ""
    st.session_state["input_2"] = ""
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# with st.form("key1"):
#     # ask for input
#     button_check = st.form_submit_button("Button to Click")

# with st.form(key="my_form", clear_on_submit=True):
#     col1, col2 = st.columns([5, 1])

    # with col1:
    #     st.text_input(
    #         "Ask me anything!",
    #         placeholder="아버지가 방에 들어가신다",
    #         key="input_1",
    #     )
    # with col2:
    #     st.text_input(
    #         "Ask me anything!",
    #         placeholder="아버지 가방에 들어가신다",
    #         key="input_2",
    #     )
#     with col3:
#         st.write("&#9660;&#9660;&#9660;")
#         submit = st.form_submit_button(label="Ask") 

msg_1 = st.text_input(label="입력 텍스트", key='msg_1')
msg_2 = st.text_input(label="대조군 텍스트", key='msg_2')

with st.form("key1"):
    # ask for input
    button_check = st.form_submit_button("Button to Click")

if button_check:
    # msg_1 = (st.session_state["input_1"], True)
    # msg_2 = (st.session_state["input_2"], True)
    # st.session_state.messages.append(msg_1)
    # st.session_state.messages.append(msg_2)

    with st.spinner("두뇌 풀가동!"):
        result = get_prediction(msg_1, msg_2)
    st.write(result)
    # msg = (result, False)
    # st.session_state.messages.append(msg)
    # message(msg[0], is_user=msg[1])

# if submit:
#     msg_1 = (st.session_state["input_1"], True)
#     msg_2 = (st.session_state["input_2"], True)
#     st.session_state.messages.append(msg_1)
#     st.session_state.messages.append(msg_2)

#     with st.spinner("두뇌 풀가동!"):
#         result = get_prediction(msg_1[0], msg_2[0])

#     msg = (result, False)
#     st.session_state.messages.append(msg)
#     message(msg[0], is_user=msg[1])
