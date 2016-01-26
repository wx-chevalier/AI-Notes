/*!
 * delete <https://github.com/jonschlinkert/delete>
 *
 * Copyright (c) 2014 Jon Schlinkert, contributors.
 * Licensed under the MIT license.
 */

'use strict';

var isPathCwd = require('is-path-cwd');
var isPathInCwd = require('is-path-in-cwd');
var rimraf = require('rimraf');

function safeCheck(filepath) {
  if (isPathCwd(filepath)) {
    throw new Error('Cannot delete the current working directory. Can be overriden with the `force` option.');
  }

  if (!isPathInCwd(filepath)) {
    throw new Error('Cannot delete files/folders outside the current working directory. Can be overriden with the `force` option.');
  }
}

module.exports = function (filepath, options, next) {
  if (typeof options === 'function') {
    next = options;
    options = {};
  }

  if (!(options && options.force)) {
    safeCheck(filepath);
  }
  return rimraf(filepath, next);
};


module.exports.sync = function (filepath, options) {
  if (!(options && options.force)) {
    safeCheck(filepath);
  }
  return rimraf.sync(filepath);
};
