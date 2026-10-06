# Changelog

## 0.6.0

- `clear({ domain })` deletes one domain only; `clear()` still clears the whole tenant.

## 0.5.0

- Requests go to `/v1/tenants/{tenantId}/…`.
- `tenantId` and `forTenant()` are the preferred way to name a tenant.
- `userId` and `forUser()` still work. When both ids are set, `tenantId` wins.
- Missing tenant now throws `tenantId is required`.
