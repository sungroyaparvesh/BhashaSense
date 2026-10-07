export type EntityCategory =
  | 'PERSON'
  | 'ORGANIZATION'
  | 'LOCATION'
  | 'PRODUCT'
  | 'MONEY'
  | 'DATE_TIME'
  | 'ISSUE_TYPE'
  | 'OTHER';

export interface ExtractedEntity {
  id: string;
  text: string;
  category: EntityCategory;
  start_char: number;
  end_char: number;
  confidence: number;
  segment_id?: number;
  start_time?: number;
  end_time?: number;
  context_sentence?: string;
}

export interface EntityGroupSummary {
  category: EntityCategory;
  count: number;
  items: string[];
}
