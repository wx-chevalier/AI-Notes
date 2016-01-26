/*!
 * is-plain-object <https://github.com/jonschlinkert/is-plain-object>
 *
 * Copyright (c) 2014 Jon Schlinkert, contributors.
 * Licensed under the MIT License
 */

'use strict';

var should = require('should');
var isPlainObject = require('./');

describe('.isPlainObject()', function () {
  it('should return `true` if the object is created by the `Object` constructor.', function () {
    isPlainObject(Object.create({})).should.be.true;
    isPlainObject(Object.create(Object.prototype)).should.be.true;
    isPlainObject({foo: 'bar'}).should.be.true;
    isPlainObject({}).should.be.true;
  });

  it('should return `false` if the object is not created by the `Object` constructor.', function () {
    function Foo() {this.abc = {};};

    isPlainObject(1).should.be.false;
    isPlainObject(['foo', 'bar']).should.be.false;
    isPlainObject([]).should.be.false;
    isPlainObject(new Foo).should.be.false;
    isPlainObject(null).should.be.false;
    isPlainObject(Object.create(null)).should.be.false;
  });
});
