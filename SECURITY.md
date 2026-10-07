# Security policy

Please do not include personal data, secrets, private documents, model files, conversation history, or absolute machine paths in issues or pull requests.

The supported security boundary is the user-facing Python desktop application and its protected `SolomonPocketAIData/` workspace. Report suspected boundary escapes, unintended network access, unsafe file parsing, or persistent camera access privately through the contact channel on [solomonchrist.com](https://www.solomonchrist.com).

Before contributing, run:

```powershell
powershell -NoProfile -File .\scripts\check-public-repo.ps1
```
