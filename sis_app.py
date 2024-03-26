import streamlit as st

from snowflake.snowpark.context import get_active_session
from snowflake.ml.registry import Registry
from snowflake.snowpark import functions as F


def get_model_version():
    reg = Registry(session=session, schema_name="MODELS")
    m = reg.get_model("DIAMONDS")
    return m.default


def get_feedback_metrics(mv) -> tuple:
    metrics = mv.show_metrics()
    review_count = metrics.get("User Rating Count", 0)
    review_counter = metrics.get("User Rating Counter", 0)
    return review_count, review_counter


def set_feedback_metrics(mv) -> None:
    mv.set_metric("User Rating Count", st.session_state.review_count)
    mv.set_metric("User Rating Counter", st.session_state.review_counter)


def vote(vote_up) -> None:
    st.session_state.review_count += 1
    if vote_up:
        st.session_state.review_counter += 1
    elif vote_up is False:
        if st.session_state.review_counter > 0:
            st.session_state.review_counter -= 1
    set_feedback_metrics(mv)
    st.session_state.voted = True


@st.cache_resource()
def get_input_options():
    t = session.table("DIAMONDS.DIAMONDS")

    cut_options = [x.CUT for x in t.select("CUT").distinct().to_local_iterator()]
    color_options = [x.COLOR for x in t.select("COLOR").distinct().to_local_iterator()]
    clarity_options = [
        x.CLARITY for x in t.select("CLARITY").distinct().to_local_iterator()
    ]

    return cut_options, color_options, clarity_options


def get_input_schema():
    t = session.table("DIAMONDS.DIAMONDS")
    return t.drop("PRICE").schema


@st.cache_data(show_spinner=False)
def get_prediction(input_data: list) -> None:
    with st.spinner("Calculating the price."):
        df = session.create_dataframe(
            [input_data],
            schema=get_input_schema(),
        )
        reg = Registry(session=session, schema_name="MODELS")
        m = reg.get_model("DIAMONDS")
        mv = m.default
        predictions = mv.run(function_name="predict", X=df)
        st.session_state.predicted_price = round(
            predictions.select("OUTPUT_PRICE").collect()[0].OUTPUT_PRICE, 2
        )
        st.session_state.voted = False


if "voted" not in st.session_state:
    st.session_state.voted = False


st.title("Diamond Price Estimator 💎")

session = get_active_session()

mv = get_model_version()

st.session_state.review_count, st.session_state.review_counter = get_feedback_metrics(
    mv
)

with st.form("Inputs"):
    cut_options, color_options, clarity_options = get_input_options()
    c1, c2, c3 = st.columns(3)
    carat_input = c2.number_input("Carat")
    cut_input = c1.selectbox("Cut", cut_options)
    color_input = c1.selectbox("Color", color_options)
    clarity_input = c1.selectbox("Clarity", clarity_options)
    depth_input = c2.number_input("Depth")
    table_input = c2.number_input("Table")
    x_input = c3.number_input("X")
    y_input = c3.number_input("Y")
    z_input = c3.number_input("Z")

    input_data = [
        carat_input,
        cut_input,
        color_input,
        clarity_input,
        depth_input,
        table_input,
        x_input,
        y_input,
        z_input,
    ]

    st.form_submit_button("Submit", on_click=get_prediction, args=(input_data,))

if "predicted_price" in st.session_state:
    st.metric(
        "Predicted Price",
        value="${:,.2f}".format(st.session_state.predicted_price),
    )

    if st.session_state.voted is False:
        st.divider()
        st.caption("Did this model meet your expectations?")
        c1, c2, c3 = st.columns([1, 1, 10])
        vote_up = c1.button("👍", on_click=vote, args=(True,))
        vote_down = c2.button("👎", on_click=vote, args=(False,))

if st.session_state.voted:
    st.divider()
    st.write(
        "Thanks for voting! 🎉 If you would like to vote again, please resubmit the form."
    )
    pct_rating = round(
        (st.session_state.review_counter / st.session_state.review_count) * 100, 2
    )
    st.info(
        f"This model has a recieved a total of {st.session_state.review_count} reviews and has a {pct_rating}% user approval rating."
    )
