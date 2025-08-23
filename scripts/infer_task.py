class InferTask:
    def __init__(self, id, tokens, max_tokens, temperature, topk, topp, end_tokens):
        self.id = id
        self.finish_reason = None
        self.tokens = tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.topk = topk
        self.topp = topp
        self.end_tokens = end_tokens
        self._kv_cache = None
        self.pos = 0

    def bind_kvcache(self, kv_cache, pos=0):
        self._kv_cache = kv_cache
        self.pos = pos
        self.tokens = self.tokens[pos:]

    def release_kvcache(self):
        cache = self._kv_cache
        self._kv_cache = None
        return cache

    def kvcache(self):
        return self._kv_cache

    def next(self, out_token):
        self._kv_cache.update_tokens(self.tokens, self.pos)
        recentWindow = 16
        self.pos += min(len(self.tokens), recentWindow)

        self.pos += len(self.tokens)
        if out_token == None or out_token in self.end_tokens:
            self.finish_reason = "stop"
        elif self.pos >= self.max_tokens:
            self.finish_reason = "length"
        else:
            self.tokens = [out_token]


class KVCache:
    def __init__(self, model):
        self._kvcache = model.create_kv_cache()
        self.tokens = [0 for _ in range(model.max_context_len())]

    def data(self):
        return self._kvcache

    def drop(self, model):
        model.drop_kv_cache(self._kvcache)

    def update_tokens(self, tokens, pos):
        end = pos + len(tokens)
        max_len = len(self.tokens)

        # If overflow, truncate tokens to fit
        if end > max_len:
            tokens = tokens[: max_len - pos]
            end = max_len

        self.tokens[pos:end] = tokens
