recommendations = {
    "Diabetes": [
        "Exercise daily for 30 minutes",
        "Follow a low-sugar, high-fiber diet",
        "Check blood sugar levels regularly",
        "Consult a diabetologist"
    ],
    "Hypothyroidism": [
        "Take thyroid hormone replacement (levothyroxine)",
        "Check TSH levels periodically",
        "Maintain a balanced diet with iodine",
        "Consult an endocrinologist"
    ],
    # ... add other diseases here
}

def get_recommendations(disease):
    normalized = disease.strip().lower()
    for key in recommendations:
        if key.lower() == normalized:
            return recommendations[key]
    return ["Consult a specialist for further diagnosis."]
