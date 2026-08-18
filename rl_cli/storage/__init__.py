"""Where this project touches the filesystem: private files, archives, caches.

Every write of a secret, a sample or a report goes through here, which is
what makes owner-only permissions and symlink refusal one policy rather than
per-call-site care. It holds no domain rules and draws nothing, so it may
read nothing from this package but itself.
"""
