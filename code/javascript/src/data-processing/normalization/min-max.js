// @flow

export function batchMinMaxNormalize(
  arr: Array<number>,
  lowerBound: number,
  upperBound: number
) {
  // 计算数组的上下界
  let min = Math.min(...arr);

  let max = Math.max(...arr);

  return arr.map(num => {
    return minMaxNormalize(num, min, max, lowerBound, upperBound);
  });
}

/**
 * Description 计算指定当前最大值、最小值以及目标上下界情况下某个值的归一化结果
 * @param value
 * @param min
 * @param max
 * @param lowerBound
 * @param upperBound
 * @return {number}
 */
export function minMaxNormalize(
  value: number,
  min: number,
  max: number,
  lowerBound: number,
  upperBound: number
) {
  return (value - min) / (max - min) * (upperBound - lowerBound) + lowerBound;
}
