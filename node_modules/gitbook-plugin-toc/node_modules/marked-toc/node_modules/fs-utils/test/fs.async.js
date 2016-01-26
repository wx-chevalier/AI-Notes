/**
 * fs-utils <https://github.com/assemble/fs-utils>
 *
 * Copyright (c) 2014 Jon Schlinkert, Brian Woodward, contributors.
 * Licensed under the MIT license.
 */

'use strict';

var should = require('should');
var path = require('path');
var file = require('..');

describe('fs', function () {
  it('should read the file (async)', function (done) {
    file.readFile('test/fixtures/test.txt', function (err, contents) {
      contents.should.eql('FILE CONTENTS!!!');
      done();
    });
  });

  it('should read JSON (async)', function (done) {
    file.readJSON('test/fixtures/test.json', function (err, contents) {
      contents.should.eql({foo: {bar: "baz"} });
      done();
    });
  });

  it('should read the yaml file (async)', function (done) {
    file.readYAML('test/fixtures/test.yaml', function (err, contents) {
      contents.should.eql({foo: {bar: "baz"}});
      done();
    });
  });

  it('should read detect JSON extension automatically (async)', function (done) {
    file.readData('test/fixtures/test.json', function (err, actual) {
      actual.should.eql({foo: {bar: "baz"} });
      done();
    });
  });

  it('should read the yaml file automatically (async)', function (done) {
    file.readData('test/fixtures/test.yaml', function (err, actual) {
      actual.should.eql({foo: {bar: "baz"}});
      done();
    });
  });

  it('should make a directory, asynchronously', function(done) {
    file.mkdir('test/actual/new/folder/async', function(err) {
      if (err) return console.log(err);
      file.exists('test/actual/new/folder/async').should.be.true;
      done();
    });
  });

  it('should remove a directory, asynchronously', function(done) {
    var existingDir = ('test/actual/new/folder/async');
    file.rmdir(existingDir, function(err) {
      if (err) return console.log(err);
      file.exists('test/actual/new/folder/async').should.be.false;
      done();
    });
  });


  it('should write a file (async)', function (done) {
    file.writeFile('test/actual/test.txt', 'FILE CONTENTS!!!', function () {
      file.readFile('test/actual/test.txt', function (err, actual) {
        file.del('test/actual/test.txt');
        actual.should.eql('FILE CONTENTS!!!');
        done();
      });
    });
  });


  it('should write JSON (async)', function (done) {
    var expected = {foo: {bar: "baz"} };
    file.writeJSON('test/actual/test.json', expected, function () {
      file.readJSON('test/actual/test.json', function (err, actual) {
        file.del('test/actual/test.json');
        actual.should.eql(expected);
        done();
      });
    });
  });

  it('should write the yaml file (async)', function (done) {
    var expected = {foo: {bar: "baz"}};
    file.writeYAML('test/actual/test.yaml', expected, function () {
      file.readYAML('test/actual/test.yaml', function (err, actual) {
        file.del('test/actual/test.yaml');
        actual.should.eql(expected);
        done();
      });
    });
  });

  it('should write JSON automatically (async)', function (done) {
    var expected = {foo: {bar: "baz"} };
    file.writeData('test/actual/test.json', expected, function () {
      file.readData('test/actual/test.json', function (err, actual) {
        file.del('test/actual/test.json');
        actual.should.eql(expected);
        done();
      });
    });
  });

  it('should write the yaml file automatically (async)', function (done) {
    var expected = {foo: {bar: "baz"}};
    file.writeData('test/actual/test.yaml', expected, function () {
      file.readData('test/actual/test.yaml', function (err, actual) {
        file.del('test/actual/test.yaml');
        actual.should.eql(expected);
        done();
      });
    });
  });

});