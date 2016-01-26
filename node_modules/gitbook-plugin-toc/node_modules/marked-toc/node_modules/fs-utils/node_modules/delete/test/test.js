/*!
 * delete <https://github.com/jonschlinkert/delete>
 *
 * Copyright (c) 2014 Jon Schlinkert, contributors.
 * Licensed under the MIT License
 */

var fs = require('fs');
var expect = require('chai').expect;
var write = require('write');
var del = require('..');


beforeEach(function () {
  write.sync('test/fixtures/a.txt', 'test');
});

describe('delete:', function () {
  describe('async:', function () {
    it('should delete files asynchronously.', function () {
      var fixture = 'test/fixtures/a.txt';
      expect(fs.existsSync(fixture)).to.be.true;
      del('test/fixtures', function(err) {
        if (err) {throw err;}
        expect(fs.existsSync(fixture)).to.be.false;
      });
    });
  });

  describe('sync:', function () {
    it('should delete files synchronously.', function () {
      var fixture = 'test/fixtures/a.txt';
      expect(fs.existsSync(fixture)).to.be.true;
      del.sync('test/fixtures');
      expect(fs.existsSync(fixture)).to.be.false;
    });
  });
});