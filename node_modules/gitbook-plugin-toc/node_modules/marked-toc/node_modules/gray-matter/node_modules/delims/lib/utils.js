/*!
 * delims <https://github.com/jonschlinkert/delims>
 *
 * Copyright (c) 2014 Jon Schlinkert
 * Licensed under the MIT license.
 */

'use strict';

var merge = require('mixin-deep');
var arrayify = require('arrayify-compact');


// Escape custom template delimiters
exports.escapeDelim = function (re) {
  return re.replace(/(.)/g, '\\$1');
};


// Build RegExp patterns for delimiters
exports.buildRegexGroup = function (re, options) {
  var opts = merge({
    escape: false,    // Escape delimiter regex
    noncapture: false // Build a non-capture group
  }, options);

  re = arrayify(re);
  var len = re.length;

  re = (len > 0) ? re.join('|') : re;
  re = (opts.escape === true)
    ? exports.escapeDelim(re)
    : re;

  if(opts.noncapture === true || len > 1) {
    re = '(?:' + re + ')';
  }
  return re;
};
