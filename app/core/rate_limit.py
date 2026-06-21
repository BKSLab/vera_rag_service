from slowapi import Limiter
from slowapi.util import get_remote_address

# API-2/SEC-2 (AUDIT_VERIFICATION_AND_IMPLEMENTATION_PLAN.md) — без rate
# limiting в сочетании с ARCH-1 (теперь закрыт API-ключом) не было вообще
# никакого барьера против шторма запросов — ни злонамеренного, ни
# случайного (например, легитимный, но забагованный клиент в retry-цикле).
# По IP, не по API-ключу — ограничение должно срабатывать даже если ключ
# уже скомпрометирован/угадан, а не только защищать от его перебора.
limiter = Limiter(key_func=get_remote_address)
