# Security policy

Use GitHub private vulnerability reporting for issues that could expose Headless
Host credentials, private room links, player information, or host/browser control.
If private reporting is unavailable, open a minimal public issue that contains no
sensitive value and ask the maintainer to establish a private channel.

Never commit Headless Host tokens, authenticated URLs, cookies, `.env` files, or
browser profiles. If a token is exposed, revoke it immediately, remove it from the
working tree, audit Git history, and sanitize the affected history before any
public push. Removing a value only from the latest file is insufficient.
