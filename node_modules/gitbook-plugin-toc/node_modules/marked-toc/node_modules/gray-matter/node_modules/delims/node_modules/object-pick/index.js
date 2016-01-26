/*!
 * object-pick <https://github.com/jonschlinkert/object-pick>
 *
 * Copyright (c) 2014 Jon Schlinkert, contributors.
 * Licensed under the MIT License
 */

'use strict';

module.exports = function pick(orig, keys) {
  if (orig == null) {
    return {};
  }

  if (typeof keys === 'string') {
    keys = [].slice.call(arguments, 1);
  }

  var len = keys.length;
  var o = {};

  for (var i = 0; i < len; i++) {
    var key = keys[i];

    if (orig.hasOwnProperty(key)) {
      o[key] = orig[key];
    }
  }
  return o;
};
