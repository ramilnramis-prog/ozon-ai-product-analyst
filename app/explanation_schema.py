EXPLANATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
        },
        "strengths": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "risks": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "next_checks": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    },
    "required": [
        "summary",
        "strengths",
        "risks",
        "next_checks",
    ],
    "additionalProperties": False,
}