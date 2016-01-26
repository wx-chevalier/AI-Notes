'use strict';

// faster slice
var slice = require('array-slice');

/**
 * Extend the target `obj` with the properties of other objects.
 *
 * @param  {Object}  `obj` The target object. Pass an empty object to shallow clone.
 * @param  {Objects}
 * @return {Object}
 */

module.exports = function extend(o) {
  if (o == null) {
    return {};
  }

  var args = slice(arguments, 1);
  var len = args.length;
  var i = 0;

  if (len === 0) {
    return o;
  }

  while (len--) {
    var obj = args[i++];

    for (var key in obj) {
      if (obj.hasOwnProperty(key)) {
        o[key] = obj[key];
      }
    }
  }
  return o;
};
