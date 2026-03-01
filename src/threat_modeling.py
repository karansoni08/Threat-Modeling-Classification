import pandas as pd


def map_threat_attributes(label_series):
    """
    Maps attack labels to structured threat modeling attributes.
    """

    threat_map = {
        "BENIGN": {
            "Vector": "None",
            "Impact": "None",
            "Control": "None",
            "Actor": "None",
            "Risk": "Low",
        },
        "DDoS": {
            "Vector": "Network Flooding",
            "Impact": "Service Disruption",
            "Control": "Rate Limiting / Firewall",
            "Actor": "External Botnet",
            "Risk": "Critical",
        },
        "PortScan": {
            "Vector": "Reconnaissance",
            "Impact": "Information Disclosure",
            "Control": "IDS / Network Monitoring",
            "Actor": "External Attacker",
            "Risk": "Medium",
        },
        "Infiltration": {
            "Vector": "Unauthorized Access",
            "Impact": "System Compromise",
            "Control": "Access Control / Patch Management",
            "Actor": "Advanced Persistent Threat",
            "Risk": "High",
        },
        "Web Attack": {
            "Vector": "Application Exploit",
            "Impact": "Data Breach",
            "Control": "Web Application Firewall",
            "Actor": "Skilled Attacker",
            "Risk": "High",
        },
    }

    vectors = []
    impacts = []
    controls = []
    actors = []
    risks = []

    for label in label_series:
        label_clean = label.strip()

        if label_clean in threat_map:
            entry = threat_map[label_clean]
        else:
            entry = {
                "Vector": "Unknown",
                "Impact": "Unknown",
                "Control": "Manual Review",
                "Actor": "Unknown",
                "Risk": "Medium",
            }

        vectors.append(entry["Vector"])
        impacts.append(entry["Impact"])
        controls.append(entry["Control"])
        actors.append(entry["Actor"])
        risks.append(entry["Risk"])

    return pd.DataFrame({
        "Threat_Vector": vectors,
        "Threat_Impact": impacts,
        "Suggested_Control": controls,
        "Threat_Actor": actors,
        "Risk_Ranking": risks
    })
