# Security Policy

## Our Commitment

We take the security of FluxPipeline seriously. This document outlines our security practices and how to report vulnerabilities.

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          | Status |
| ------- | ------------------ | ------ |
| 0.1.x   | :white_check_mark: | Active development |
| < 0.1   | :x:                | Not supported |

**Note**: This project is currently in early development (v0.1.x). We recommend always using the latest version.

## Known Security Considerations

### AI Model Usage

FluxPipeline uses the FLUX.1-schnell model for image generation. Users should be aware:

1. **Generated Content**: The model generates images based on text prompts. Users are responsible for:
   - Ensuring prompts comply with applicable laws
   - Not generating harmful, illegal, or inappropriate content
   - Following the model's usage restrictions

2. **Model Provenance**: Always download models from official sources:
   - Official: [HuggingFace - black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
   - Verify model checksums when possible

3. **Content Filtering**: This project does not include content filtering by default. Implement your own safety measures if deploying in production.

### Data Privacy

1. **Local Processing**: All image generation happens locally. No data is sent to external servers unless you explicitly configure cloud storage or APIs.

2. **Prompt Storage**: Generation prompts are stored locally in:
   - `workspace/history.json` (if using the GUI)
   - Log files (if logging is enabled)

   These files may contain sensitive information. Protect them accordingly.

3. **Environment Variables**: Sensitive configuration (API tokens, credentials) should be stored in `.env` files, which are gitignored by default.

### Dependency Security

We regularly update dependencies to patch security vulnerabilities:

- **Automated Scanning**: GitHub Actions run weekly security scans
- **Dependency Review**: All dependency updates are reviewed for security implications
- **Known Vulnerabilities**: Check [GitHub Security Advisories](https://github.com/Xza85hrf/flux_pipeline/security/advisories)

### GPU and System Access

FluxPipeline requires significant system access for GPU operations:

1. **CUDA/ROCm Access**: Direct GPU access for NVIDIA/AMD cards
2. **Memory Management**: Can allocate large amounts of GPU/system memory
3. **File System**: Creates files in workspace directory

**Security Best Practice**: Run in a containerized environment (Docker) with appropriate resource limits when possible.

## Reporting a Vulnerability

### How to Report

We appreciate responsible disclosure of security vulnerabilities. Please **DO NOT** create public GitHub issues for security vulnerabilities.

Instead, report vulnerabilities via:

1. **GitHub Security Advisories** (preferred)
   - Go to: https://github.com/Xza85hrf/flux_pipeline/security/advisories
   - Click "Report a vulnerability"
   - Provide detailed information

2. **Email** (if GitHub is unavailable)
   - Email project maintainers through GitHub
   - Use subject line: "[SECURITY] FluxPipeline Vulnerability Report"

### What to Include

Please provide as much information as possible:

```markdown
## Vulnerability Description
Brief description of the issue

## Affected Components
- Component/file affected
- Version(s) affected
- Configuration (if specific)

## Impact
- Severity (Critical/High/Medium/Low)
- Potential impact (data loss, code execution, etc.)
- Attack vector (local/remote, authentication required, etc.)

## Reproduction Steps
1. Step one
2. Step two
3. See vulnerability

## Proof of Concept
Code/commands to reproduce (if applicable)

## Suggested Fix
If you have suggestions for remediation

## Discoverer
Your name/handle (for credit)
```

### What to Expect

After you submit a vulnerability report:

1. **Acknowledgment**: We'll acknowledge receipt within **48 hours**
2. **Assessment**: We'll assess severity and validity within **5 business days**
3. **Communication**: We'll keep you informed of our progress
4. **Resolution Timeline**:
   - **Critical**: Patch within 7 days
   - **High**: Patch within 30 days
   - **Medium**: Patch within 90 days
   - **Low**: Patch in next regular release

5. **Disclosure**: We'll coordinate public disclosure with you
6. **Credit**: You'll be credited in the security advisory (unless you prefer to remain anonymous)

### Out of Scope

The following are generally considered out of scope:

- Issues in third-party dependencies (report to upstream projects)
- Social engineering attacks
- Physical attacks
- DoS attacks that require significant resources
- Issues affecting unsupported versions
- Issues requiring physical access to the machine
- Theoretical vulnerabilities without proof of concept

## Security Best Practices for Users

### For Development

1. **Use Virtual Environments**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Keep Dependencies Updated**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

3. **Scan for Vulnerabilities**
   ```bash
   pip install safety
   safety check
   ```

4. **Use Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

### For Production Deployment

⚠️ **Warning**: This project is NOT recommended for production use in its current state. It's designed for educational and experimental purposes.

If you must deploy in production:

1. **Containerization**
   - Use Docker with resource limits
   - Run as non-root user
   - Use read-only file systems where possible

2. **Network Isolation**
   - Don't expose Gradio interface to public internet without authentication
   - Use VPN or firewall rules
   - Consider adding authentication layer (nginx reverse proxy with auth)

3. **Resource Limits**
   ```yaml
   # docker-compose.yml example
   services:
     flux:
       deploy:
         resources:
           limits:
             memory: 16G
             cpus: '8'
   ```

4. **Monitoring**
   - Monitor resource usage
   - Set up alerts for anomalies
   - Log all generation requests

5. **Content Moderation**
   - Implement prompt filtering
   - Log all prompts for review
   - Consider content safety APIs

6. **API Security**
   - Use HTTPS only
   - Implement rate limiting
   - Add authentication/authorization
   - Validate all inputs

### For Model Downloads

1. **Verify Sources**
   ```bash
   # Only download from official HuggingFace
   # Verify with checksums if available
   ```

2. **Use Hugging Face Authentication**
   ```bash
   # Set token for private models
   export HF_TOKEN="your_token_here"
   # Or use huggingface-cli login
   ```

3. **Scan Downloaded Models**
   - Check file sizes match expected
   - Verify no unexpected files
   - Use virus/malware scanners

## Security Features

### Current Protections

1. **Dependency Pinning**: All dependencies are pinned to specific versions
2. **Input Validation**: Basic validation of generation parameters
3. **Error Handling**: Proper exception handling to prevent crashes
4. **Resource Cleanup**: Automatic GPU memory cleanup
5. **Gitignore**: Sensitive files excluded from version control

### Planned Enhancements

- [ ] Content safety filtering
- [ ] Rate limiting for API/GUI
- [ ] Audit logging
- [ ] Encrypted storage for sensitive data
- [ ] Security headers for web interface

## Vulnerability Disclosure Policy

### Coordinated Disclosure

We practice coordinated vulnerability disclosure:

1. Security researcher reports vulnerability privately
2. We work together to understand and fix the issue
3. We prepare and test a patch
4. We coordinate public disclosure timing
5. Credit is given to the researcher (if desired)

### Public Disclosure Timeline

- **For critical vulnerabilities**: Disclosed after patch is available
- **For other vulnerabilities**: 90-day disclosure deadline from report
- **Extensions**: Available if needed to ensure proper fixes

## Security Hall of Fame

We recognize security researchers who help make FluxPipeline more secure:

<!-- This section will be updated as vulnerabilities are reported and fixed -->

*No vulnerabilities reported yet.*

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE Top 25](https://cwe.mitre.org/top25/)
- [HuggingFace Security](https://huggingface.co/docs/hub/security)
- [PyTorch Security](https://pytorch.org/docs/stable/security.html)

## Contact

For security-related questions that aren't vulnerabilities:
- Create a GitHub Discussion with the "security" tag
- Email maintainers through GitHub

---

**Thank you for helping keep FluxPipeline and its users safe!** 🔒
