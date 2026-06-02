# QR Code Generator Prototype

## System Requirements

Build a dynamic QR code system where:
- Users submit a long URL and get back a short URL token + QR code image
- The QR code encodes a short URL that redirects (302) to the original URL via your server
- Users can modify the target URL after QR code creation
- Users can delete a QR code (soft delete)
- Users can optionally set an expiration timestamp on create or update
- Deleted or expired links return appropriate HTTP status codes
- URL validation: format check, normalization, malicious URL blocking

## Design Questions

Answer these before you start coding:

1. **Static vs Dynamic QR Code:** Why does this system use dynamic QR codes (encode short URL) instead of static (encode original URL directly)? When would you choose static instead?
- Dynamic (encode the short URL) because the QR image becomes immutable once printed/shared — encoding the original URL directly would mean every URL change requires regenerating and redistributing the QR. The short URL gives you an indirection layer: change the target, keep the same QR. You also get analytics (every scan hits your server) and the ability to revoke/expire links.
- Choose static when: the destination will truly never change (e.g. a mailto:, a Wi-Fi password, a vCard, a payment URI), you want zero server dependency / offline scanning, or privacy requires that scans not be logged. Static is also faster (no redirect hop) and survives your service going down.

2. **Token Generation:** How will you generate short URL tokens? What happens when two different URLs produce the same token? How does collision probability change as the number of tokens grows?
- Random base62 with 8 chars,  Insert into a DB column with a UNIQUE constraint on token. On collision, retry up to 5 times. Don't dedupe by URL — same URL from two users gets two different tokens.                                         

3. **Redirect Strategy:** Why 302 (temporary) instead of 301 (permanent)? What are the trade-offs for analytics, URL modification, and latency?
- temporary redirect is preferred if we want to get analytical insights from the users, to check how many times a certain QR is accessed. 
- Redirecting may incur longer load times particularly if its being requested by many users at the same time, can consider to add caching with TTL to reduce the latency

4. **URL Normalization:** What normalization rules do you need? Why is `http://Example.com/` and `https://example.com` potentially the same URL?
- Rules I would apply before storing or comparing:
1) lowercase the scheme and host as path/query is case sensitive
2) strip the default port (80 for http, 443 for https)
3) resolve . and .. in the path
4) Add a trailing slash on bare hosts (example.com → example.com/)

5. **Error Semantics:** What should happen when someone scans a deleted link vs a non-existent link? Should the HTTP status codes be different?
- scanning a deleted link: 410 Gone
- scanning a non existent link: 404 Error as resource does not exist

## Verification

Your prototype should pass all of these:

```bash
# Create a QR code
curl -X POST http://localhost:8000/api/qr/create \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
# → 200, returns {"token": "...", "short_url": "...", "qr_code_url": "...", "original_url": "..."}

# Redirect
curl -o /dev/null -w "%{http_code}" http://localhost:8000/r/{token}
# → 302

# Get info
curl http://localhost:8000/api/qr/{token}
# → 200, returns token metadata

# Update target URL
curl -X PATCH http://localhost:8000/api/qr/{token} \
  -H "Content-Type: application/json" \
  -d '{"url": "https://new-url.com"}'
# → 200

# Redirect now goes to new URL
curl -o /dev/null -w "%{redirect_url}" http://localhost:8000/r/{token}
# → https://new-url.com

# Delete
curl -X DELETE http://localhost:8000/api/qr/{token}
# → 200

# Redirect after delete
curl -o /dev/null -w "%{http_code}" http://localhost:8000/r/{token}
# → 410

# Non-existent token
curl -o /dev/null -w "%{http_code}" http://localhost:8000/r/INVALID
# → 404

# QR code image
# (create a new one first, then)
curl -o /dev/null -w "%{http_code} %{content_type}" http://localhost:8000/api/qr/{token}/image
# → 200 image/png

# Analytics
curl http://localhost:8000/api/qr/{token}/analytics
# → 200, returns {"token": "...", "total_scans": N, "scans_by_day": [...]}
```

## Suggested Tech Stack

Python + FastAPI recommended, but you may use any language/framework.
