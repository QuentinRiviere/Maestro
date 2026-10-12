import {
  Book, Camera, Clapperboard, Film, Gamepad2, Heart, Megaphone, Mic, Music, Palette, Sparkles, Zap,
  type LucideIcon,
} from 'lucide-react'

export const SKILL_ICONS: Record<string, LucideIcon> = {
  sparkles: Sparkles, film: Film, music: Music, mic: Mic, camera: Camera, clapperboard: Clapperboard,
  megaphone: Megaphone, book: Book, gamepad: Gamepad2, palette: Palette, heart: Heart, zap: Zap,
}

export function skillIcon(name?: string): LucideIcon {
  return SKILL_ICONS[name || ''] || Sparkles
}
