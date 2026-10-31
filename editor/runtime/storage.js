// This binding is private to the modularized engine factory. The existing C
// persistence helpers use it instead of the normal game's storage namespace.
var localStorage = {
  getItem: function(key) { return window.localStorage.getItem('ab3d1.editor.' + key); },
  setItem: function(key, value) { window.localStorage.setItem('ab3d1.editor.' + key, value); },
  removeItem: function(key) { window.localStorage.removeItem('ab3d1.editor.' + key); }
};
