/*!
 * kind-of <https://github.com/jonschlinkert/kind-of>
 *
 * Copyright (c) 2014 Jon Schlinkert, contributors.
 * Licensed under the MIT License
 */

'use strict';

var should = require('should');
var kindOf = require('./');

describe('kindOf', function () {
  describe('nulls', function () {
    it('should work for undefined', function () {
      kindOf(undefined).should.equal('undefined');
    });

    it('should work for null', function () {
      kindOf(null).should.equal('null');
    });
  });

  describe('primitives', function () {
    it('should work for booleans', function () {
      kindOf(true).should.equal('boolean');
      kindOf(false).should.equal('boolean');
      kindOf(new Boolean(true)).should.equal('boolean');
    });

    it('should work for numbers', function () {
      kindOf(42).should.equal('number');
      kindOf(new Number(42)).should.equal('number');
    });

    it('should work for strings', function () {
      kindOf("string").should.equal('string');
    });
  });

  describe('objects', function () {
    it('should work for arguments', function () {
      (function () {
        kindOf(arguments).should.equal('arguments');
        return;
      })();
    });

    it('should work for objects', function () {
      function Test() {}
      kindOf({}).should.equal('object');
      kindOf(new Test()).should.equal('object');
    });

    it('should work for dates', function () {
      kindOf(new Date()).should.equal('date');
    });

    it('should work for arrays', function () {
      kindOf([]).should.equal('array');
      kindOf([1, 2, 3]).should.equal('array');
      kindOf(new Array()).should.equal('array');
    });

    it('should work for regular expressions', function () {
      kindOf(/[\s\S]+/).should.equal('regexp');
      kindOf(new RegExp('^' + 'foo$')).should.equal('regexp');
    });

    it('should work for functions', function () {
      kindOf(function () {}).should.equal('function');
      kindOf(new Function()).should.equal('function');
    });
  });
});