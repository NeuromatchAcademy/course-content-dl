# Stub ipywidgets for headless/CI execution.
# Replaces blocking widget calls with no-ops so notebooks execute without hanging.
# In Colab/Jupyter with a real frontend, the real ipywidgets is used instead.
#
# Installed into ~/.ipython/profile_default/startup/ by the setup-ci-tools action
# so it runs automatically before any notebook cell when nbconvert spawns a kernel.
import sys
import types
import inspect


class _NoOpWidget:
    """A no-op stand-in for any ipywidgets widget class."""

    children = []

    def __init__(self, *args, **kwargs):
        # Preserve value/options so _Interact can extract call defaults
        object.__setattr__(self, "value", kwargs.get("value", None))
        object.__setattr__(self, "options", kwargs.get("options", []))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        # Return a no-op callable for any unknown method/attribute
        return lambda *args, **kwargs: None


class _Interact:
    """Stub for widgets.interact / widgets.interactive.

    Calls the wrapped function once with default values extracted from
    widget stubs so that matplotlib outputs are captured by nbconvert.
    """

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            # Bare @widgets.interact — extract defaults from widget params
            return self._call_with_defaults(args[0])
        # @widgets.interact(param=slider) — return decorator
        widget_kwargs = kwargs

        def decorator(f):
            return self._call_with_defaults(f, widget_kwargs)

        return decorator

    def _call_with_defaults(self, f, widget_kwargs=None):
        sig = inspect.signature(f)
        call_kwargs = {}
        for name, param in sig.parameters.items():
            widget = (widget_kwargs or {}).get(name)
            if widget is None and param.default is not inspect.Parameter.empty:
                widget = param.default
            if isinstance(widget, _NoOpWidget) and widget.value is not None:
                call_kwargs[name] = widget.value
            elif widget is not None and not isinstance(widget, _NoOpWidget):
                call_kwargs[name] = widget
        try:
            f(**call_kwargs)
        except Exception as e:
            print(f"[stub] interact call skipped: {e}")
        return f


class _StubModule(types.ModuleType):
    """ipywidgets stub module.

    Any attribute access returns _NoOpWidget so that
    'from ipywidgets import AnythingAtAll' always succeeds.
    """

    interact = _Interact()
    interactive = _Interact()

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _NoOpWidget


stub = _StubModule("ipywidgets")
stub.widgets = stub  # support: from ipywidgets import widgets

# --- Output widget stub that captures display() ---
class _NoOpOutput:
    """Stub for widgets.Output that captures display() calls."""
    def __init__(self):
        self.children = []
        self._output_buffer = []

    def __enter__(self):
        _current_output_ref[0] = self
        return self

    def __exit__(self, *args):
        # Store captured output in children so nbconvert picks it up
        if self._output_buffer:
            self.children = self._output_buffer
        _current_output_ref[0] = None

    def __repr__(self):
        return f"<_NoOpOutput children={self.children}>"


_current_output_ref = [None]


def _patched_display(*objs, **kwargs):
    """Capture display() calls into the current Output widget."""
    current = _current_output_ref[0]
    if current is not None:
        current._output_buffer.extend(objs)
    # Also call the real display so IPython renders it
    try:
        from IPython import display as ipy_display
        ipy_display.display(*objs, **kwargs)
    except Exception:
        pass


# Register Output class on the stub
stub.Output = _NoOpOutput

sys.modules["ipywidgets"] = stub
sys.modules["ipywidgets.widgets"] = stub

# Patch IPython.display.display to capture outputs
try:
    from IPython import display as ipy_display
    _original_display = ipy_display.display
    ipy_display.display = _patched_display
    print("IPython.display.display patched for output capture")
except Exception:
    pass

print("ipywidgets stubbed for headless CI execution")

# --- tqdm stub for headless execution ---
# In headless CI, tqdm.notebook crashes because the Jupyter widget
# container has no children. Force tqdm to use the standard
# terminal implementation instead.
try:
    import tqdm.std
    sys.modules["tqdm.notebook"] = tqdm.std
    sys.modules["tqdm._tqdm_notebook"] = tqdm.std
    print("tqdm.notebook stubbed: using std implementation")
except Exception:
    pass

# --- Resilient URL fetching for CI ---
# Two problems with fetching remote data (e.g. OSF) from CI on Python 3.10:
#
# 1. 308 redirects: Python 3.10's urllib doesn't support 308 (added in 3.11).
#    OSF uses 308 redirects. http_error_308 alone is not enough — 3.10's
#    HTTPRedirectHandler.redirect_request raises HTTPError for any code not in
#    {301, 302, 303, 307}, so we also remap 308 -> 307 in redirect_request
#    (both preserve the request method) so the redirect is actually followed.
#
# 2. Transient 5xx: OSF / its CDN occasionally returns 502/503/504, and bare
#    connection errors happen. A single failed fetch flakes the whole CI run,
#    so we wrap the opener to retry these with exponential backoff.
#
# TODO: Remove the 308 handler once CI runs on Python 3.11+ (308 is native there).
#       The retry wrapper is still useful regardless of Python version.
import time
import urllib.request
import urllib.error

_RETRY_STATUSES = {502, 503, 504}
_MAX_RETRIES = 3          # total attempts = _MAX_RETRIES + 1
_BACKOFF_BASE = 1.0       # seconds; delay = _BACKOFF_BASE * 2**attempt


class _HTTP308Handler(urllib.request.HTTPRedirectHandler):
    """Handle 308 Permanent Redirect by following the Location header."""

    def http_error_308(self, req, fp, code, msg, headers):
        return self.http_error_302(req, fp, code, msg, headers)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # 3.10's redirect_request rejects 308 outright; treat it as 307.
        if code == 308:
            code = 307
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _install_resilient_opener():
    opener = urllib.request.build_opener(_HTTP308Handler)
    _inner_open = opener.open

    def _open_with_retry(fullurl, data=None,
                         timeout=urllib.request.socket._GLOBAL_DEFAULT_TIMEOUT):
        last_exc = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return _inner_open(fullurl, data, timeout)
            except urllib.error.HTTPError as e:
                if e.code not in _RETRY_STATUSES or attempt == _MAX_RETRIES:
                    raise
                last_exc = e
                reason = f"HTTP {e.code}"
            except urllib.error.URLError as e:
                # Connection-level failure (DNS, refused, reset, timeout).
                if attempt == _MAX_RETRIES:
                    raise
                last_exc = e
                reason = f"URLError: {e.reason}"
            delay = _BACKOFF_BASE * (2 ** attempt)
            url_str = getattr(fullurl, "full_url", fullurl)
            print(
                f"[stub] fetch failed ({reason}) for {url_str} — "
                f"retry {attempt + 1}/{_MAX_RETRIES} in {delay:.1f}s"
            )
            time.sleep(delay)
        # Unreachable, but keep the contract explicit.
        raise last_exc

    opener.open = _open_with_retry
    urllib.request.install_opener(opener)


_install_resilient_opener()
print("resilient URL opener installed (308 redirects + 5xx retry)")
