'use strict';

var isObject = require('is-plain-object');
var forOwn = require('for-own');

module.exports = function deepMixin(o) {
  var args = [].slice.call(arguments, 1);
  var len = args.length;

  if (o == null) {
    return {};
  }

  if (len === 0) {
    return o;
  }

  function copy(value, key) {
    var obj = this[key];
    if (isObject(value) && isObject(obj)) {
      deepMixin(obj, value);
    } else {
      this[key] = value;
    }
  }

  for (var i = 0; i < len; i++) {
    var obj = args[i];
    if (obj != null) {
      forOwn(obj, copy, o);
    }
  }
  return o;
};
