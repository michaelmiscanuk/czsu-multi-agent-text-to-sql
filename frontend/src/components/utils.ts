// Utility to remove diacritics from a string (Czech and other languages)
export function removeDiacritics(str: string): string {
  return str.normalize('NFD').replace(/\p{Diacritic}/gu, '');
} 