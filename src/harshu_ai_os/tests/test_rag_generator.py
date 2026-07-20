from harshu_ai_os.rag.generator import create_rag_generator


def test_create_rag_generator_passes_route_and_prompt():
    route = {
        "model": "fake-model",
        "max_tokens": 200,
    }

    received = {}

    def fake_llm_call(received_route, received_prompt):
        received["route"] = received_route
        received["prompt"] = received_prompt
        return "Grounded test answer"

    generate_text = create_rag_generator(
        route,
        llm_call=fake_llm_call,
    )

    answer = generate_text("Use this retrieved context.")

    assert answer == "Grounded test answer"
    assert received["route"] == route
    assert received["prompt"] == "Use this retrieved context."