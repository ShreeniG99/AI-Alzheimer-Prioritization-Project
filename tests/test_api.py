STAGE1 = {"age": 74, "education": 14, "sex": 2, "mmse": 23, "adas": 20, "faq": 8}
PLASMA = {"ptau217": 0.9, "ab42_ab40": 0.085, "nfl": 25, "gfap": 160, "apoe4": 1}
MRI = {"eTIV": 1500, "nWBV": 0.71, "ASF": 1.17, "hippocampus_icv": 0.0045}
PET = {"amyloid_centiloid": 80, "fdg": 1.15}


def test_health_is_public_and_lists_stages(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["available_stages"] == ["stage1", "stage2", "stage3", "stage4"]


def test_protected_endpoints_require_token(client):
    assert client.get("/model-info").status_code == 401
    assert client.post("/assessment", json=STAGE1).status_code == 401
    bad = {"Authorization": "Bearer not-a-token"}
    assert client.get("/patients", headers=bad).status_code == 401


def test_wrong_password_is_rejected_and_audited(client, admin_headers):
    assert client.post("/auth/login", json={"username": "admin", "password": "nope"}).status_code == 401
    actions = [e["action"] for e in client.get("/audit", headers=admin_headers).json()]
    assert "login_failed" in actions


def test_model_info_reports_all_stages(client, admin_headers):
    info = client.get("/model-info", headers=admin_headers).json()
    assert [s["stage"] for s in info["stages"]] == ["stage1", "stage2", "stage3", "stage4"]
    assert all(s["available"] for s in info["stages"])
    assert info["stages"][0]["cohort"]["participants"] > 0


def test_assessment_scores_explains_and_saves(client, admin_headers):
    response = client.post("/assessment", headers=admin_headers, json={"patient_ref": "t-1", **STAGE1})
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["prediction"]["current_stage"] == "stage1"
    assert 0 <= body["prediction"]["risk_score"] <= 1
    assert body["prediction"]["priority"] in {"LOW", "MEDIUM", "HIGH"}
    assert body["explanation"]["method"].startswith("SHAP")
    labels = {c["feature"] for c in body["explanation"]["contributions"]}
    assert {"age", "mmse", "adas", "faq", "education", "sex"} <= labels
    assert body["data_quality"]["level"] == "GOOD"
    assert body["assessment_id"].startswith("ASM-")

    saved = client.get(f"/patients/{body['id']}", headers=admin_headers).json()
    assert saved["risk_score"] == body["prediction"]["risk_score"]
    assert saved["explanation"]["contributions"]


def test_duplicate_assessment_id_conflicts(client, admin_headers):
    payload = {"assessment_id": "ASM-DUPLICATE", **STAGE1}
    assert client.post("/assessment", headers=admin_headers, json=payload).status_code == 201
    assert client.post("/assessment", headers=admin_headers, json=payload).status_code == 409


def test_cdr_zero_routes_to_early_detection(client, admin_headers):
    body = client.post("/assessment", headers=admin_headers, json={**STAGE1, "mmse": 29, "cdr": 0}).json()
    assert body["task"] == "early_detection"


def test_pathway_progresses_through_all_stages(client, admin_headers):
    created = client.post("/assessment", headers=admin_headers, json={"patient_ref": "t-path", **STAGE1}).json()
    record_id = created["id"]

    for values, stage in [(PLASMA, "stage2"), (MRI, "stage3"), (PET, "stage4")]:
        response = client.patch(f"/patients/{record_id}", headers=admin_headers,
                                json={"values": values, "capacity": 2})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["task"] == "rerank"
        assert body["patient"]["current_stage"] == stage
        assert body["queue"]["selected_count"] == 2

    trace = body["patient"]["pathway"]["trace"]
    assert [t["stage"] for t in trace] == ["stage1", "stage2", "stage3", "stage4"]
    assert body["patient"]["next_step"]["next_stage"] is None


def test_triage_respects_capacity_and_order(client, admin_headers):
    body = client.post("/triage", headers=admin_headers, json={"capacity": 2}).json()
    risks = [p["risk_score"] for p in body["patients"]]
    assert risks == sorted(risks, reverse=True)
    assert [p["selected_for_capacity"] for p in body["patients"]][:2] == [True, True]
    assert sum(p["selected_for_capacity"] for p in body["patients"]) == 2


def test_rank_adhoc_cohort(client, admin_headers):
    body = client.post("/rank", headers=admin_headers, json={
        "patients": [{**STAGE1, "mmse": 29, "adas": 5, "faq": 0}, {**STAGE1, "mmse": 18, "adas": 30, "faq": 20}],
        "capacity": 1,
    }).json()
    assert body["patients"][0]["selected_for_capacity"]
    assert body["patients"][0]["risk_score"] >= body["patients"][1]["risk_score"]


def test_lower_cognition_raises_risk(client, admin_headers):
    def risk(**overrides):
        payload = {**STAGE1, **overrides, "stage": "stage1"}
        return client.post("/predict", headers=admin_headers, json=payload).json()["risk_score"]

    assert risk(mmse=18, adas=35, faq=20) > risk(mmse=30, adas=3, faq=0)


def test_uncertainty_unmatched_patient_is_not_extrapolated(client, admin_headers):
    body = client.post("/uncertainty", headers=admin_headers, json={**STAGE1, "adas": 12.3456}).json()
    assert body["available"] is False
    assert 0 <= body["production_calibrated_risk"] <= 1


def test_input_validation(client, admin_headers):
    assert client.post("/assessment", headers=admin_headers, json={"mmse": 45}).status_code == 422


def test_delete_patient(client, admin_headers):
    record_id = client.post("/assessment", headers=admin_headers, json=STAGE1).json()["id"]
    assert client.delete(f"/patients/{record_id}", headers=admin_headers).status_code == 204
    assert client.get(f"/patients/{record_id}", headers=admin_headers).status_code == 404


def test_admin_only_routes(client, admin_headers):
    created = client.post("/auth/users", headers=admin_headers,
                          json={"username": "dr.test", "password": "password123", "role": "clinician"})
    assert created.status_code == 201
    token = client.post("/auth/login", json={"username": "dr.test", "password": "password123"}).json()["access_token"]
    clinician = {"Authorization": f"Bearer {token}"}
    assert client.get("/audit", headers=clinician).status_code == 403
    assert client.post("/auth/users", headers=clinician,
                       json={"username": "x-user", "password": "password123"}).status_code == 403
    assert client.get("/patients", headers=clinician).status_code == 200


def test_genotype_alone_does_not_reach_stage2(client, admin_headers):
    body = client.post("/assessment", headers=admin_headers, json={**STAGE1, "apoe4": 2}).json()
    assert body["prediction"]["current_stage"] == "stage1"
    assert body["pathway"]["stages_reached"] == ["stage1"]
