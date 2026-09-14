# Corporate Acceptable Use and Remote Access Governance Policy

## 1. Remote Access and Virtual Private Network Standards
All employee remote access to corporate engineering assets, private repositories, and internal production clusters must be routed through enterprise zero-trust network access (ZTNA) gateways requiring hardware-token Multi-Factor Authentication (MFA). Direct SSH or RDP exposure to the public internet is categorically forbidden.

## 2. Credential Management and Secret Governance
Under no circumstances may engineers embed API keys, database credentials, SSH private keys, or personal access tokens directly into code repositories, pull requests, or issue trackers. All operational secrets must be fetched at runtime from the enterprise secret manager or injected via masked CI/CD environment variables.

## 3. Generative AI and External LLM Usage Boundaries
Employees are prohibited from submitting proprietary source code, unreleased intellectual property, or confidential customer Personally Identifiable Information (PII) to public, unapproved consumer AI services. All internal AI workflows must operate through the vetted enterprise private AI gateway with mandatory PII redaction and audit logging enabled.
