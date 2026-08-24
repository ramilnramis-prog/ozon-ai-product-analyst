CATEGORY_CLASSIFICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "category_type": {
            "type": "string",
            "enum": [
                "homogeneous",
                "mixed",
                "unknown",
            ],
        },
        "functional_families": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                    },
                    "keywords": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                },
                "required": [
                    "name",
                    "keywords",
                ],
                "additionalProperties": False,
            },
        },
        "confidence": {
            "type": "number",
        },
    },
    "required": [
        "category_type",
        "functional_families",
        "confidence",
    ],
    "additionalProperties": False,
}

FAMILY_RESOLUTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "resolutions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                    },
                    "family_name": {
                        "type": "string",
                    },
                    "confidence": {
                        "type": "number",
                    },
                },
                "required": [
                    "product_name",
                    "family_name",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "resolutions",
    ],
    "additionalProperties": False,
}