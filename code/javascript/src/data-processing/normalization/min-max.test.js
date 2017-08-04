// @flow
import { batchMinMaxNormalize, minMaxNormalize } from './min-max';

describe('测试 MinMax 归一化', () => {
  it('batchMinMaxNormalize', () => {
    expect(batchMinMaxNormalize([-99, 8, 10, 15, 20], 1, 100)).toHaveLength(5);
  });

  it('minMaxNormalize 正确转换', () => {
    expect(minMaxNormalize(8, 8, 20, 0, 1)).toEqual(0);

    // 这里需要进行四舍五入
    expect(minMaxNormalize(10, 8, 20, 0, 1)).toBeCloseTo(0.17, 2);

    expect(minMaxNormalize(20, 8, 20, 0, 1)).toEqual(1);
  });
});
